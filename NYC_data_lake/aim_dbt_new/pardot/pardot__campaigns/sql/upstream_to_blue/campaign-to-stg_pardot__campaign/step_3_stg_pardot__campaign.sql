with base as (

    select * 
    from "pardot"."public_stg_pardot"."stg_pardot__campaign_tmp"

),

fields as (

    select
        
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    cost
    
 as 
    
    cost
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    



        
    from base
    where not coalesce(_fivetran_deleted, false)
),

final as (
    
    select 
        id as campaign_id,
        name as campaign_name,
        cost,
        _fivetran_deleted,
        _fivetran_synced
    from fields
)

select * from final