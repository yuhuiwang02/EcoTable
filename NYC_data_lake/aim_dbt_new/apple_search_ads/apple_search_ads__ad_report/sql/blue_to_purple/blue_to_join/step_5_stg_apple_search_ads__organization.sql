

with base as (

    select * 
    from "apple_search_ads"."public_apple_search_ads_dev"."stg_apple_search_ads__organization_tmp"
),

fields as (

    select
        
    
    
    currency
    
 as 
    
    currency
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    payment_model
    
 as 
    
    payment_model
    
, 
    
    
    time_zone
    
 as 
    
    time_zone
    



        
    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        id as organization_id,
        currency,
        payment_model,
        name as organization_name,
        time_zone
    from fields
)

select * 
from final