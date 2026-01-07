with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__page_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    app_id
    
 as 
    
    app_id
    
, 
    
    
    color
    
 as 
    
    color
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    created_by_user_id
    
 as 
    
    created_by_user_id
    
, 
    
    
    dirty
    
 as 
    
    dirty
    
, 
    
    
    group_id
    
 as 
    
    group_id
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    last_updated_at
    
 as 
    
    last_updated_at
    
, 
    
    
    last_updated_by_user_id
    
 as 
    
    last_updated_by_user_id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    root_version_id
    
 as 
    
    root_version_id
    
, 
    
    
    stable_version_id
    
 as 
    
    stable_version_id
    
, 
    
    
    valid_through
    
 as 
    
    valid_through
    



        
    from base
),

final as (
    
    select 
        id as page_id,
        name as page_name,
        app_id,
        created_at,
        created_by_user_id,
        dirty as is_dirty,
        group_id,
        last_updated_at,
        last_updated_by_user_id,
        root_version_id,
        stable_version_id,
        cast(valid_through as timestamp) as valid_through,
        _fivetran_synced

    from fields
)

select * 
from final