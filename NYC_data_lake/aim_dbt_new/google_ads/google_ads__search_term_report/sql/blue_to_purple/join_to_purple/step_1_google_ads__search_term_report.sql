

with stats as (

    select *
    from "google_ads"."public_google_ads_dev"."stg_google_ads__search_term_keyword_stats"
), 

accounts as (

    select *
    from "google_ads"."public_google_ads_dev"."stg_google_ads__account_history"
    where is_most_recent_record = True
),

campaigns as (

    select *
    from "google_ads"."public_google_ads_dev"."stg_google_ads__campaign_history"
    where is_most_recent_record = True
), 

ad_groups as (

    select *
    from "google_ads"."public_google_ads_dev"."stg_google_ads__ad_group_history"
    where is_most_recent_record = True
), 

fields as (

    select
        stats.source_relation,
        stats.date_day,
        accounts.account_name,
        stats.account_id,
        accounts.currency_code,
        campaigns.campaign_name,
        stats.campaign_id,
        ad_groups.ad_group_name,
        stats.ad_group_id,
        stats.search_term,
        stats.keyword_text,
        stats.criterion_id,
        stats.search_term_match_type,
        stats.status,
        sum(stats.spend) as spend,
        sum(stats.clicks) as clicks,
        sum(stats.impressions) as impressions,
        sum(stats.conversions) as conversions,
        sum(stats.conversions_value) as conversions_value,
        sum(stats.view_through_conversions) as view_through_conversions

        





    from stats
    left join ad_groups
        on stats.ad_group_id = ad_groups.ad_group_id
        and stats.source_relation = ad_groups.source_relation
    left join campaigns
        on stats.campaign_id = campaigns.campaign_id
        and stats.source_relation = campaigns.source_relation
    left join accounts
        on stats.account_id = accounts.account_id
        and stats.source_relation = accounts.source_relation
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14
)

select *
from fields