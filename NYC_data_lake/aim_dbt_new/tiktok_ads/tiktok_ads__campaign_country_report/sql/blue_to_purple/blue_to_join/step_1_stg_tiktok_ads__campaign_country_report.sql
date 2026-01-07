

with base as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__campaign_country_report_tmp"
), 

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    clicks
    
 as 
    
    clicks
    
, 
    
    
    conversion
    
 as 
    
    conversion
    
, 
    
    
    conversion_rate
    
 as 
    
    conversion_rate
    
, 
    
    
    cost_per_conversion
    
 as 
    
    cost_per_conversion
    
, 
    
    
    country_code
    
 as 
    
    country_code
    
, 
    
    
    cpc
    
 as 
    
    cpc
    
, 
    
    
    cpm
    
 as 
    
    cpm
    
, 
    
    
    ctr
    
 as 
    
    ctr
    
, 
    
    
    impressions
    
 as 
    
    impressions
    
, 
    
    
    real_time_conversion
    
 as 
    
    real_time_conversion
    
, 
    
    
    spend
    
 as 
    
    spend
    
, 
    
    
    stat_time_day
    
 as 
    
    stat_time_day
    



    
        


, cast('' as TEXT) as source_relation




    from base
), 

final as (

    select
        campaign_id,
        cast(stat_time_day as date) as stat_time_day,
        country_code,
        coalesce(clicks, 0) as clicks,
        coalesce(impressions, 0) as impressions,
        coalesce(spend, 0) as spend,
        coalesce(conversion, 0) as conversion,
        coalesce(conversion_rate, 0) as conversion_rate,
        coalesce(cost_per_conversion, 0) as cost_per_conversion,
        coalesce(cpc, 0) as cpc,
        coalesce(cpm, 0) as cpm,
        coalesce(ctr, 0) as ctr,
        coalesce(real_time_conversion, 0) as real_time_conversion,
        source_relation,
        _fivetran_synced

        





    from fields
)

select *
from final