

with base as (

    select * 
    from "google_ads"."public_google_ads_dev"."stg_google_ads__campaign_stats_tmp"
),

fields as (

    select
        
    cast(null as TEXT) as 
    
    _fivetran_id
    
 , 
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as TEXT) as 
    
    ad_network_type
    
 , 
    cast(null as integer) as 
    
    clicks
    
 , 
    cast(null as integer) as 
    
    cost_micros
    
 , 
    cast(null as integer) as 
    
    customer_id
    
 , 
    cast(null as date) as 
    
    date
    
 , 
    cast(null as TEXT) as 
    
    device
    
 , 
    cast(null as integer) as 
    
    id
    
 , 
    cast(null as integer) as 
    
    impressions
    
 , 
    cast(null as integer) as 
    
    conversions
    
 , 
    cast(null as integer) as 
    
    conversions_value
    
 , 
    cast(null as integer) as 
    
    view_through_conversions
    
 , 
    cast(null as TEXT) as view_through_conversions_aliased 


    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        customer_id as account_id, 
        date as date_day, 
        id as campaign_id, 
        ad_network_type,
        device,
        coalesce(clicks, 0) as clicks, 
        coalesce(cost_micros, 0) / 1000000.0 as spend, 
        coalesce(impressions, 0) as impressions,
        coalesce(conversions, 0) as conversions,
        coalesce(conversions_value, 0) as conversions_value,
        coalesce(view_through_conversions, 0) as view_through_conversions
        
        


    
        
    
        
            
                , coalesce(cast(view_through_conversions_aliased as float), 0) as view_through_conversions_aliased
            
        
    




    from fields
)

select *
from final