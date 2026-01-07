

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__search_term_report_tmp"
),

fields as (

    select
        
    
    
    _fivetran_id
    
 as 
    
    _fivetran_id
    
, 
    
    
    ad_group_deleted
    
 as 
    
    ad_group_deleted
    
, 
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    ad_group_name
    
 as 
    
    ad_group_name
    
, 
    
    
    bid_amount_amount
    
 as 
    
    bid_amount_amount
    
, 
    
    
    bid_amount_currency
    
 as 
    
    bid_amount_currency
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    date
    
 as 
    
    date
    
, 
    
    
    deleted
    
 as 
    
    deleted
    
, 
    
    
    conversions
    
 as 
    
    conversions
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    
    
    keyword
    
 as 
    
    keyword
    
, 
    
    
    keyword_display_status
    
 as 
    
    keyword_display_status
    
, 
    
    
    keyword_id
    
 as 
    
    keyword_id
    
, 
    
    
    local_spend_amount
    
 as 
    
    local_spend_amount
    
, 
    
    
    local_spend_currency
    
 as 
    
    local_spend_currency
    
, 
    
    
    match_type
    
 as 
    
    match_type
    
, 
    
    
    new_downloads
    
 as 
    
    new_downloads
    
, 
    
    
    redownloads
    
 as 
    
    redownloads
    
, 
    
    
    search_term_source
    
 as 
    
    search_term_source
    
, 
    
    
    search_term_text
    
 as 
    
    search_term_text
    
, 
    
    
    taps
    
 as 
    
    taps
    
, 
    
    
    tap_installs
    
 as 
    
    tap_installs
    
, 
    
    
    tap_new_downloads
    
 as 
    
    tap_new_downloads
    
, 
    
    
    tap_redownloads
    
 as 
    
    tap_redownloads
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        date as date_day,
        _fivetran_id,
        campaign_id,
        ad_group_id,
        ad_group_name,
        bid_amount_amount as bid_amount,
        bid_amount_currency as bid_currency,
        keyword as keyword_text,
        keyword_display_status,
        keyword_id,
        local_spend_amount as spend,
        local_spend_currency as currency,
        match_type,
        search_term_source,
        search_term_text,
        impressions,
        taps,
        coalesce(conversions, tap_installs) as conversions, 
        coalesce(tap_installs, conversions) as tap_installs,
        coalesce(new_downloads, tap_new_downloads) as new_downloads,
        coalesce(tap_new_downloads, new_downloads) as tap_new_downloads,
        coalesce(redownloads, tap_redownloads) as redownloads,
        coalesce(tap_redownloads, redownloads) as tap_redownloads

        





    from fields
)

select * 
from final