with base as (

    select * 
    from "pardot"."public_stg_pardot"."stg_pardot__list_membership_tmp"

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
    
    
    id
    
 as 
    
    id
    
, 
    
    
    list_id
    
 as 
    
    list_id
    
, 
    
    
    opted_out
    
 as 
    
    opted_out
    
, 
    
    
    prospect_id
    
 as 
    
    prospect_id
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    



        
    from base
),

final as (
    
    select 
        id as list_membership_id,
        prospect_id,
        list_id,
        created_at as created_timestamp,
        updated_at as updated_timestamp,
        opted_out as has_opted_out,
        _fivetran_synced
    from fields
)

select * from final