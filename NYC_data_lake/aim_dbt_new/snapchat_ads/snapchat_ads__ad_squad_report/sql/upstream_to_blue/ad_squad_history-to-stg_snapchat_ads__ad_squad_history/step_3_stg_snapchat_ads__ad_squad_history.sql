

with base as (

    select * 
    from "snapchat_ads"."public_snapchat_ads_dev"."stg_snapchat_ads__ad_squad_history_tmp"
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
    cast(null as timestamp) as 
    
    created_at
    
 , 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    cast(null as timestamp) as 
    
    updated_at
    
 


    
        


, cast('' as TEXT) as source_relation




    from base
),

final as (

    select
        source_relation, 
        id as ad_squad_id,
        name as ad_squad_name,
        cast (created_at as timestamp) as created_at,
        campaign_id,
        cast (_fivetran_synced as timestamp) as _fivetran_synced,
        cast (updated_at as timestamp) as updated_at,
        row_number() over (partition by source_relation, id order by _fivetran_synced desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final