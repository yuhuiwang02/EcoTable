with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__feature_event_tmp"

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
    
    
    feature_id
    
 as 
    
    feature_id
    
, 
    
    
    num_events
    
 as 
    
    num_events
    
, 
    
    
    num_minutes
    
 as 
    
    num_minutes
    
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
        feature_id,
        num_events,
        num_minutes,
        remote_ip,
        server_name,
        timestamp as occurred_at,
        user_agent,
        visitor_id,
        _fivetran_synced,
        _fivetran_id,
        md5(cast(coalesce(cast(visitor_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(timestamp as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(account_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(server_name as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(feature_id as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(remote_ip as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(user_agent as TEXT), '_dbt_utils_surrogate_key_null_') || '-' || coalesce(cast(_fivetran_id as TEXT), '_dbt_utils_surrogate_key_null_') as TEXT)) as feature_event_key

        --The below macro adds the fields defined within your pendo__feature_event_pass_through_columns variable into the staging model
        




        
    from fields
)

select * 
from final