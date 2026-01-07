

with country_report as (
    
    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__campaign_country_report"
), 

campaigns as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__campaign_history"
    where is_most_recent_record
), 

advertiser as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__advertiser"
), 

aggregated as (

    select
        country_report.source_relation,
        country_report.stat_time_day as date_day,
        country_report.campaign_id,
        campaigns.campaign_name,
        campaigns.campaign_type,
        campaigns.created_at,
        country_report.country_code,
        advertiser.advertiser_id,
        advertiser.advertiser_name,
        advertiser.currency,
        campaigns.objective_type,
        campaigns.status,
        campaigns.split_test_variable,
        campaigns.budget,
        campaigns.budget_mode,
        sum(country_report.clicks) as clicks,
        sum(country_report.impressions) as impressions,
        sum(country_report.spend) as spend,
        sum(country_report.conversion) as conversion,
        sum(country_report.spend)/nullif(sum(country_report.clicks),0) as daily_cpc,
        (sum(country_report.spend)/nullif(sum(country_report.impressions),0))*1000 as daily_cpm,
        (sum(country_report.clicks)/nullif(sum(country_report.impressions),0))*100 as daily_ctr,
        sum(country_report.real_time_conversion) as real_time_conversion

        



        

    from country_report
    left join campaigns
        on country_report.campaign_id = campaigns.campaign_id
        and country_report.source_relation = campaigns.source_relation
    left join advertiser
        on campaigns.advertiser_id = advertiser.advertiser_id
        and campaigns.source_relation = advertiser.source_relation

    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
)

select *
from aggregated