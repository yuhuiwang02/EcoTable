

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__ad_report_tmp"
),

fields as (

    select
        
    
    
    ad_group_id
    
 as 
    
    ad_group_id
    
, 
    
    
    ad_id
    
 as 
    
    ad_id
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
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
    
    
    conversions
    
 as conversions_alias_should_be_included 


    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        date as date_day,
        campaign_id,
        ad_group_id,
        ad_id,
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

        


    
        
            
                , coalesce(cast(conversions_alias_should_be_included as float), 0) as conversions_alias_should_be_included
            
        
    




    from fields
)

select *
from final