with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__group_tmp"

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
    
    
    description
    
 as 
    
    description
    
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
    
    
    length
    
 as 
    
    length
    
, 
    
    
    name
    
 as 
    
    name
    



        
    from base
),

final as (
    
    select 
        id as group_id,
        app_id,
        created_at,
        created_by_user_id,
        description,
        last_updated_at,
        last_updated_by_user_id,
        name as group_name,
        _fivetran_synced

    from fields
)

select * 
from final