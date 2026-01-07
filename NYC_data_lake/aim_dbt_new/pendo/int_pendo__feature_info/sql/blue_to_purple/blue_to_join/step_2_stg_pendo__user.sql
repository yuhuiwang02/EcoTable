with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__user_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    deleted_at
    
 as 
    
    deleted_at
    
, 
    
    
    first_name
    
 as 
    
    first_name
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    last_name
    
 as 
    
    last_name
    
, 
    
    
    role
    
 as 
    
    role
    
, 
    
    
    user_type
    
 as 
    
    user_type
    
, 
    
    
    username
    
 as 
    
    username
    



        
    from base
),

final as (
    
    select 

        id as user_id,
        deleted_at,
        first_name,
        last_name,
        user_type,
        username,
        _fivetran_synced

    from fields
)

select * 
from final