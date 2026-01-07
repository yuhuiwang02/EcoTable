{{ config(enabled=fivetran_utils.enabled_vars(['ad_reporting__apple_search_ads_enabled','apple_search_ads__using_search_terms'])) }}

with report as (

    select *
    from {{ ref('stg_apple_search_ads__search_term_report') }}
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
        report.campaign_id, 
        campaign.campaign_name, 
        report.ad_group_id,
        report.ad_group_name,
        report.keyword_id,
        report.keyword_text,
        report.search_term_text,
        report.match_type,
        report.currency,
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
            pass_through_variable = 'apple_search_ads__search_term_passthrough_metrics',
            identifier = 'report',
            transform = 'sum',
            coalesce_with = 0,
            exclude_fields = ['conversions']) }}

    from report
    left join campaign 
        on report.campaign_id = campaign.campaign_id
        and report.source_relation = campaign.source_relation
    left join organization 
        on campaign.organization_id = organization.organization_id
        and campaign.source_relation = organization.source_relation
    where report.search_term_text is not null
    {{ dbt_utils.group_by(13) }}
)

select * 
from joined
