import os
import json
import copy
import logging
import sys
import pandas as pd
import numpy as np

from pycarol import ApiKeyAuth, PwdAuth, PwdKeyAuth, Staging, DataModel, Query, CDSGolden
from pycarol import Storage, Connectors, CarolAPI, Carol, CarolHandler, Apps
from pycarol import CDSGolden, CDSStaging
from pycarol.utils.miscellaneous import drop_duplicated_parquet
from pycarol import _CAROL_METADATA_UNTIE_GOLDEN

from pytechfin import Techfin
from pytechfin import CarolSyncMonitoring
from pytechfin.enums import EnumApps

from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
login = Carol()
carol = CarolHandler(login)
carol.setLevel(logging.INFO)
logger.addHandler(carol)

app = Apps(login)
settings = app.get_settings(app_name='datadecorationstats')
os.environ['TECHFINCLIENTID'] = settings['techfinclientid']
os.environ['TECHFINCLIENTSECRET'] = settings['techfinclientsecret']


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))
    excepthook(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception

STATS_COLUMNS = [
    'tenant',
    'data_model',
    'golden_records',
    'last_updated_golden_record',
    'rejected_records',
    'last_updated_rejected_record',
    'failed_lookup_records',
    'last_updated_failed_lookup_record',
    'techfin_records',
    'records_diff',
    'golden_record_stats',
    'subscription_id']

KEY_MAP = {
    'apinvoiceaccounting': 'invoiceaccounting_id',
    'apinvoice': 'invoice_id',
    'apinvoiceinstallment': 'invoiceinstallment_id',
    'apinvoicepayments': 'invoicepayments_id',
    'arinvoiceaccounting': 'invoiceaccounting_id',
    'arinvoicebra': 'invoicebra_id',
    'arinvoice': 'invoice_id',
    'arinvoiceinstallment': 'invoiceinstallment_id',
    'arinvoiceorigin': '_id',
    'arinvoicepartner': 'invoicepartner_id',
    'arinvoicepayments': 'invoicepayments_id',
    'arpaymentstype': 'transactiontype_id',
    'cfbankbalance': '_id',
    'company': 'uuid',
    'fndbankaccount': 'bankaccount_id',
    'mdaccount': 'account_id',
    'mdbankaccount': 'bank_id',
    'mdbusinesspartnerdocreference': '_id',
    'mdbusinesspartner': 'businesspartner_id',
    'mdbusinesspartnergroup': 'businesspartnergroup_id',
    'mdcostcenter': 'costcenter_id',
    'mdcurrency': 'currency_id',
    'mddocreference': 'docreference_id',
    'mdfinancialcategory': 'financialcategoryid',
    'organization': 'uuid'}

DATA_MODEL_MAP = {
    "fndHierarchyOrganization": "company",
    "fndOrganizations": "organization",
    "mdCurrencies": "mdcurrency",
    "mdCostCenters": "mdcostcenter",
    "mdAccounts": "mdaccount",
    "mdBankAccounts": "mdbankaccount",
    "mdBankAccountsBRA": "fndbankaccount",
    "cfBankBalances": "cfbankbalance",
    "mdPaymentTypes": "arpaymentstype",
    "mdFinancialCategories": "mdfinancialcategory",
    "mdBusinessPartnerGroups": "mdbusinesspartnergroup",
    "mdBusinessPartners": "mdbusinesspartner",
    "mdDocReferences": "mddocreference",
    "mdBusinessPartnerDocReferences": "mdbusinesspartnerdocreference",
    "arInvoices": "arinvoice",
    "arInvoiceInstallments": "arinvoiceinstallment",
    "arInvoicePayments": "arinvoicepayments",
    "arInvoiceAccountings": "arinvoiceaccounting",
    "apInvoices": "apinvoice",
    "apInvoiceInstallments": "apinvoiceinstallment",
    "apInvoicePayments": "apinvoicepayments",
    "apInvoiceAccountings": "apinvoiceaccounting"}

DATAMODEL_LIST = sorted(list(DATA_MODEL_MAP.values()), reverse=True)


def get_subscription_id(carol, data_model):
    subscription_id = 'no data subscription'
    dm_id = carol.datamodel.get_by_name(data_model)['mdmId']
    subscription = carol.call_api(
        f'v3/subscription/template/{dm_id}', method='GET')
    if len(subscription) > 0:
        subscription_id = subscription[0]['mdmId']
    return subscription_id


def datetime_logger(text, value=''):
    log = ''
    if value:
        log = f'{str(datetime.today())} - {text}: {value}'
    else:
        log = f'{str(datetime.today())} - {text}'
    logger.info(log)


def get_dm(carol, dm_name, max_workers=30, columns=None, max_hits=None, return_metadata=True, merge_records=True, file_pattern=None, callback=None):

    df = carol.datamodel.fetch_parquet(
        dm_name=dm_name,
        return_metadata=return_metadata, merge_records=merge_records,
        max_workers=max_workers,
        max_hits=max_hits, columns=columns,
        file_pattern=file_pattern
    )
    return df


def get_rejected(carol, dm_name, max_hits=None, max_workers=30, file_pattern=None, remove_duplicates=True, staging_record=False, debug=False):
    df = carol.datamodel.fetch_rejected(
        dm_name=dm_name,
        max_workers=max_workers,
        max_hits=max_hits,
        file_pattern=file_pattern
    )

    if df.empty:
        if debug:
            print(f'no rejected for {dm_name}')
        return df

    if remove_duplicates:
        if debug:
            print(f'{dm_name} before merges {df.shape}')
        df = drop_duplicated_parquet(df)

    if 'mdmErrors' in df.columns:
        size = len(df.loc[df.mdmErrors.isna()]['mdmErrors'])
        df.loc[df.mdmErrors.isna(), 'mdmErrors'] = [{} for _ in range(size)]
    else:
        if debug:
            print('missing mdmErrors column')

    if (staging_record) and ('mdmStagingRecord' in df.columns):
        size = len(df.loc[df.mdmStagingRecord.isna()]['mdmStagingRecord'])
        df.loc[df.mdmStagingRecord.isna(), 'mdmStagingRecord'] = [{}
                                                                  for _ in range(size)]
        aux = pd.DataFrame(df.pop('mdmStagingRecord').tolist())
        aux.columns = [i.replace('_string', '') for i in aux.columns]
        aux.columns = [
            i + '_from_staging' if i.startswith('mdm') else i for i in aux.columns]
        df = pd.concat([df, aux, ], axis=1)
    else:
        if debug:
            print('missing mdmStagingRecord column')

    if debug:
        print(f'{dm_name}  final: {df.shape}')
    return df


def get_dd_tenants(carol):
    tenant_list = get_dm(carol, 'caroltenant')
    tenant_list = tenant_list[(tenant_list.datadecoratio.fillna(False))][[
        'orgname', 'tenantname']]
    return tenant_list.drop_duplicates(subset=['orgname', 'tenantname']).sort_values(['orgname', 'tenantname'])


def get_filter(setting):
    filter = None
    if setting != '':
        filter = set(setting.split(','))
    return filter


def get_techfin_data(org, tenant):
    techfin_data = list()
    if (org.lower() == 'totvstechfin'):
        tf = Techfin()
        dms_techfin = CarolSyncMonitoring(tf)
        techfin_data = dms_techfin.get_table_record_count(
            techfin_app=EnumApps.CASHFLOW.value, carol_tenant=tenant)
        techfin_data = {DATA_MODEL_MAP[i['tableName']]: i['count']
                        for i in techfin_data if i['tableName'] in DATA_MODEL_MAP}
    return techfin_data


def get_data_model_data(login, tenant, data_model, techfin_data, ignorerejectedrecords=False, stats_last_hours=0, stats_trace_log=False):
    data_model_data = []
    golden_records = 0
    last_updated_golden_record = ''
    rejected_records = 0
    last_updated_rejected_record = ''
    failed_lookup_records = 0
    last_updated_failed_lookup_record = ''
    techfin_records = 0
    records_diff = 0
    subscription_id = ''
    golden_record_stats = ''

    golden_data = get_dm(login, data_model, columns=[
                         'mdmId', 'mdmLastUpdated'])
    golden_records = golden_data.mdmId.nunique()
    last_updated_golden_record = golden_data.mdmLastUpdated.max()
    if stats_trace_log:
        datetime_logger('golden_data', str(golden_records))

    subscription_id = get_subscription_id(login, data_model)
    if stats_trace_log:
        datetime_logger('subscription_id', subscription_id)

    if not(ignorerejectedrecords):
        rejected_data = get_rejected(login, data_model)
        rejected_records = rejected_data.shape[0]
        if (rejected_records > 0):
            last_updated_rejected_record = rejected_data.mdmLastUpdated.max()
            if (rejected_records > 0) and ('mdmSourceType' in rejected_data.columns):
                failed_lookup_data = rejected_data[rejected_data.mdmSourceType == 'DECORATION']
                failed_lookup_records = failed_lookup_data.shape[0]
                if (failed_lookup_records > 0):
                    last_updated_failed_lookup_record = failed_lookup_data.mdmLastUpdated.max()
            if stats_trace_log:
                datetime_logger('rejected_data', str(rejected_records))
                datetime_logger('failed_lookup_records',
                                str(failed_lookup_records))
    else:
        rejected_records = -1
        datetime_logger('rejected_data', str(rejected_records))

    if len(techfin_data) > 0:
        techfin_records = techfin_data.get(data_model, 0)
        records_diff = golden_records - techfin_records
        if stats_trace_log:
            datetime_logger('techfin_records', str(techfin_records))
            datetime_logger('records_diff', str(records_diff))

    if stats_last_hours < 0:
        filtered_data = golden_data[pd.to_datetime(
            golden_data['mdmStagingCounter']/1000, unit='ms') >= (datetime.utcnow() - timedelta(hours=stats_last_hours))]
        golden_record_stats = str((pd.to_datetime(filtered_data['mdmCounterForEntity']/1000, unit='ms') - pd.to_datetime(
            filtered_data['mdmStagingCounter']/1000, unit='ms')).describe().fillna(0)['mean'])
        if stats_trace_log:
            datetime_logger('golden_record_stats', str(golden_record_stats))

    data_model_data.append([tenant,
                            data_model,
                            golden_records,
                            last_updated_golden_record,
                            rejected_records,
                            last_updated_rejected_record,
                            failed_lookup_records,
                            last_updated_failed_lookup_record,
                            techfin_records,
                            records_diff,
                            str(golden_record_stats),
                            subscription_id])

    df = pd.DataFrame(data_model_data, columns=STATS_COLUMNS)
    return df


def sync_tenant_data(carol, data_model_data):
    staging = Staging(carol)
    staging.send_data(staging_name='tenant_stats_data',
                      data=data_model_data,
                      force=True,
                      print_stats=True,
                      async_send=True,
                      max_workers=30,
                      connector_name='datadecoration',
                      step_size=1000,
                      auto_create_schema=False,
                      crosswalk_auto_create=['tenant', 'data_model'])


def data_decoration_stats():
    datetime_logger('begin')
    carol = CarolAPI()
    tenant_list = get_dd_tenants(carol)
    dm_counter = 1

    carolorgfilter = get_filter(settings['carolorgfilter'])
    datetime_logger('carolorgfilter', carolorgfilter)
    caroltenantfilter = get_filter(settings['caroltenantfilter'])
    datetime_logger('caroltenantfilter', caroltenantfilter)
    stats_last_hours = settings['goldenrecordstatslasthours']
    datetime_logger('goldenrecordstatslasthours', str(stats_last_hours))
    statstracelog = settings['statstracelog']
    datetime_logger('statstracelog', str(statstracelog))
    ignorerejectedrecords = settings['ignorerejectedrecords']
    datetime_logger('ignorerejectedrecords', str(ignorerejectedrecords))

    for tenant in tenant_list.itertuples(index=False):
        if (carolorgfilter is None) or (len(carolorgfilter) > 0) and (tenant.orgname in carolorgfilter):
            if (caroltenantfilter is None) or (len(caroltenantfilter) > 0) and (tenant.tenantname in caroltenantfilter):
                datetime_logger(
                    'org/tenant', f'{tenant.orgname}/{tenant.tenantname}')
                with carol.switch_context(env_name=tenant.tenantname, org_name=tenant.orgname, app_name="techfinplatform") as carol_tenant:
                    techfin_data = get_techfin_data(
                        tenant.orgname, tenant.tenantname)
                    dm_counter = 1
                    for data_model in DATAMODEL_LIST:
                        datetime_logger(f'{dm_counter} - {data_model}')
                        data_model_data = get_data_model_data(
                            carol_tenant, tenant.tenantname, data_model, techfin_data, ignorerejectedrecords, stats_last_hours, statstracelog)
                        sync_tenant_data(carol, data_model_data)
                        dm_counter += 1
    datetime_logger('end')


if __name__ == '__main__':
    data_decoration_stats()
