

with report as (

    select *
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__campaign_daily_report"

), 

campaigns as (

    select *
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__campaign_history"
    where is_most_recent_record = True
),

accounts as (

    select *
    from "microsoft_ads"."public_microsoft_ads_dev"."stg_microsoft_ads__account_history"
    where is_most_recent_record = True
),

joined as (

    select
        report.source_relation,
        report.date_day,
        accounts.account_name,
        report.account_id,
        campaigns.campaign_name,
        report.campaign_id,
        campaigns.type as campaign_type,
        campaigns.time_zone as campaign_timezone,
        campaigns.status as campaign_status,
        report.device_os,
        report.device_type,
        report.network,
        report.currency_code,
        sum(report.clicks) as clicks,
        sum(report.impressions) as impressions,
        sum(report.spend) as spend,
        sum(report.conversions) as conversions,
        sum(report.conversions_value) as conversions_value,
        sum(report.all_conversions) as all_conversions,
        sum(report.all_conversions_value) as all_conversions_value

        



 

    from report
    left join accounts
        on report.account_id = accounts.account_id
        and report.source_relation = accounts.source_relation
    left join campaigns
        on report.campaign_id = campaigns.campaign_id
        and report.source_relation = campaigns.source_relation
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13
)

select *
from joined