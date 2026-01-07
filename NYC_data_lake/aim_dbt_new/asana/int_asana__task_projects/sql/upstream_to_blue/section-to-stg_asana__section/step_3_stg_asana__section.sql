with base as (

    select * 
    from "asana"."public_asana_dev"."stg_asana__section_tmp"

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
    
    
    name
    
 as 
    
    name
    
, 
    
    
    project_id
    
 as 
    
    project_id
    



        
    from base
),

final as (
    
    select 
        id as section_id,
        cast(created_at as timestamp) as created_at,
        name as section_name,
        project_id
    from fields
)

select * 
from final