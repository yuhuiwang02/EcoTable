{{ config(enabled=var('ad_reporting__apple_search_ads_enabled', True)) }}

with report as (

    select *
    from {{ ref('stg_apple_search_ads__keyword_report') }}
), 

keyword as (

    select *
    from {{ ref('stg_apple_search_ads__keyword_history') }}
    where is_most_recent_record = True
), 

ad_group as (

    select *
    from {{ ref('stg_apple_search_ads__ad_group_history') }}
    where is_most_recent_record = True
), 

campaign as (

    select *
    from {{ ref('stg_apple_search_ads__campaign_history') }}
    where is_most_recent_record = True
), 

organization as (

    select * 
    from {{ ref('stg_apple_search_ads__organization') }}
), 

joined as (

    select 
        report.source_relation,
        report.date_day,
        organization.organization_id,
        organization.organization_name,
        campaign.campaign_id, 
        campaign.campaign_name, 
        ad_group.ad_group_id,
        ad_group.ad_group_name,
        report.keyword_id,
        keyword.keyword_text,
        keyword.match_type,
        report.currency,
        keyword.keyword_status,
        sum(report.taps) as taps,
        sum(report.new_downloads) as new_downloads, -- this will be deprecated shortly; please reference tap_new_downloads instead
        sum(report.tap_new_downloads) as tap_new_downloads,
        sum(report.redownloads) as redownloads, -- this will be deprecated shortly; please reference tap_redownloads instead
        sum(report.tap_redownloads) as tap_redownloads,
        sum(report.new_downloads + report.redownloads) as total_downloads, -- this will be deprecated shortly; please reference tap_total_downloads instead
        sum(report.tap_new_downloads + report.tap_redownloads) as tap_total_downloads,
        sum(report.conversions) as conversions, -- this will be deprecated shortly; please reference tap_installs instead
        sum(report.tap_installs) as tap_installs,
        sum(report.impressions) as impressions,
        sum(report.spend) as spend

        {{ apple_search_ads_persist_pass_through_columns(
            pass_through_variable = 'apple_search_ads__keyword_passthrough_metrics',
            identifier = 'report',
            transform = 'sum',
            coalesce_with = 0,
            exclude_fields = ['conversions']) }}

    from report
    left join keyword 
        on report.keyword_id = keyword.keyword_id
        and report.source_relation = keyword.source_relation
    left join ad_group 
        on keyword.ad_group_id = ad_group.ad_group_id
        and keyword.source_relation = ad_group.source_relation
    left join campaign 
        on ad_group.campaign_id = campaign.campaign_id
        and ad_group.source_relation = campaign.source_relation
    left join organization 
        on ad_group.organization_id = organization.organization_id
        and ad_group.source_relation = organization.source_relation
    {{ dbt_utils.group_by(13) }}
)

select * 
from joined
