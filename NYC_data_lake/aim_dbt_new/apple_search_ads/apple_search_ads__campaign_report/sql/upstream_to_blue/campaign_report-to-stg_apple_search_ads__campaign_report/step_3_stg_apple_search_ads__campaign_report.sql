

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__campaign_report_tmp"
),

fields as (

    select
        
    
    
    date
    
 as 
    
    date
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    conversions
    
 as 
    
    conversions
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    
    
    local_spend_amount
    
 as 
    
    local_spend_amount
    
, 
    
    
    local_spend_currency
    
 as 
    
    local_spend_currency
    
, 
    
    
    new_downloads
    
 as 
    
    new_downloads
    
, 
    
    
    redownloads
    
 as 
    
    redownloads
    
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
        id as campaign_id,
        impressions,
        local_spend_amount as spend,
        local_spend_currency as currency,
        coalesce(conversions, tap_installs) as conversions, 
        coalesce(tap_installs, conversions) as tap_installs,
        coalesce(new_downloads, tap_new_downloads) as new_downloads,
        coalesce(tap_new_downloads, new_downloads) as tap_new_downloads,
        coalesce(redownloads, tap_redownloads) as redownloads,
        coalesce(tap_redownloads, redownloads) as tap_redownloads,
        taps

        





    from fields
)

select * 
from final