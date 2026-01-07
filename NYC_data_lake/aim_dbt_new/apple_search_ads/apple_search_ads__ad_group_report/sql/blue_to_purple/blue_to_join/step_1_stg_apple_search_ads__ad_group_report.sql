

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__ad_group_report_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    date
    
 as 
    
    date
    
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
    
, 
    
    
    lat_off_installs
    
 as 
    
    lat_off_installs
    



    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation,
        date as date_day, 
        ad_group_id,
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

        


    
        
    
        
            
                , coalesce(cast(lat_off_installs as float), 0) as lat_off_installs
            
        
    




    from fields
)

select * 
from final