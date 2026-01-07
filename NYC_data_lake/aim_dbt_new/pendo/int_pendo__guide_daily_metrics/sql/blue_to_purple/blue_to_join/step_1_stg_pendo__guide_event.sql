with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__guide_event_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    account_id
    
 as 
    
    account_id
    
, 
    
    
    app_id
    
 as 
    
    app_id
    
, 
    
    
    country
    
 as 
    
    country
    
, 
    
    
    element_path
    
 as 
    
    element_path
    
, 
    
    
    guide_id
    
 as 
    
    guide_id
    
, 
    
    
    guide_step_id
    
 as 
    
    guide_step_id
    
, 
    
    
    latitude
    
 as 
    
    latitude
    
, 
    
    
    load_time
    
 as 
    
    load_time
    
, 
    
    
    longitude
    
 as 
    
    longitude
    
, 
    
    
    region
    
 as 
    
    region
    
, 
    
    
    remote_ip
    
 as 
    
    remote_ip
    
, 
    
    
    server_name
    
 as 
    
    server_name
    
, 
    
    
    timestamp
    
 as 
    
    timestamp
    
, 
    
    
    type
    
 as 
    
    type
    
, 
    
    
    url
    
 as 
    
    url
    
, 
    
    
    user_agent
    
 as 
    
    user_agent
    
, 
    
    
    visitor_id
    
 as 
    
    visitor_id
    
, 
    
    
    _fivetran_id
    
 as 
    
    _fivetran_id
    



        
    from base
),

final as (
    
    select 
        account_id,
        app_id,
        country,
        guide_id,
        guide_step_id,
        latitude,
        longitude,
        region,
        remote_ip,
        server_name,
        timestamp as occurred_at,
        type,
        url,
        user_agent,
        visitor_id,
        _fivetran_synced,
        _fivetran_id,
        md5(cast(coalesce(cast(visitor_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(timestamp as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(account_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(server_name as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(guide_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(user_agent as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(remote_ip as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(_fivetran_id as TEXT), '_dbt_utils_surrogate_key_null_') as TEXT)) 
            as guide_event_key

    from fields
)

select * 
from final