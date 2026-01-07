{{ config(
    tags="fivetran_validations",
    enabled=var('fivetran_validation_tests_enabled', false)
) }}

with ad_source as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.stg_apple_search_ads__ad_report
),

ad_model as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.apple_search_ads__ad_report
),

ad_group_source as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.stg_apple_search_ads__ad_group_report
),

ad_group_model as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.apple_search_ads__ad_group_report
),

campaign_source as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.stg_apple_search_ads__campaign_report
),

campaign_model as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.apple_search_ads__campaign_report
),

keyword_source as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.stg_apple_search_ads__keyword_report
),

keyword_model as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.apple_search_ads__keyword_report
),

organization_model as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.apple_search_ads__organization_report
),

search_term_source as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.stg_apple_search_ads__search_term_report
    where search_term_text is not null
),

search_term_model as (

    select 
        sum(coalesce(conversions, 0)) as conversions
    from {{ target.schema }}_apple_search_ads_dev.apple_search_ads__search_term_report
)

select 
    'ads' as comparison
from ad_model 
join ad_source on true
where abs(ad_model.conversions - ad_source.conversions) >= .01
    
union all 

select 
    'organizations' as comparison
from organization_model 
join campaign_source on true
where abs(organization_model.conversions - campaign_source.conversions) >= .01

union all 

select 
    'ad groups' as comparison
from ad_group_model 
join ad_group_source on true
where abs(ad_group_model.conversions - ad_group_source.conversions) >= .01

union all 

select 
    'campaigns' as comparison
from campaign_model 
join campaign_source on true
where abs(campaign_model.conversions - campaign_source.conversions) >= .01

union all 

select 
    'keywords' as comparison
from keyword_model 
join keyword_source on true
where abs(keyword_model.conversions - keyword_source.conversions) >= .01

union all 

select 
    'search_terms' as comparison
from search_term_model 
join search_term_source on true
where abs(search_term_model.conversions - search_term_source.conversions) >= .01