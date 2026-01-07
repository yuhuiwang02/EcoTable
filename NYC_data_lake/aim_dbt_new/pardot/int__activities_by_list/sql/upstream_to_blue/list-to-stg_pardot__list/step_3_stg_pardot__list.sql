with base as (

    select * 
    from "pardot"."public_stg_pardot"."stg_pardot__list_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    description
    
 as 
    
    description
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    is_crm_visible
    
 as 
    
    is_crm_visible
    
, 
    
    
    is_dynamic
    
 as 
    
    is_dynamic
    
, 
    
    
    is_public
    
 as 
    
    is_public
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    title
    
 as 
    
    title
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    



        
    from base
),

final as (
    
    select 
        id as list_id,
        name,
        description,
        title,
        is_crm_visible,
        is_public,
        is_dynamic,
        created_at as created_timestamp,
        updated_at as updated_timestamp,
        _fivetran_synced
    from fields

)

select * from final