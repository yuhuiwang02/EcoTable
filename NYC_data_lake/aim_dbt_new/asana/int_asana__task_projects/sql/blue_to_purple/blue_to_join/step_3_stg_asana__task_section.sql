with base as (

    select * 
    from "asana"."public_asana_dev"."stg_asana__task_section_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    section_id
    
 as 
    
    section_id
    
, 
    
    
    task_id
    
 as 
    
    task_id
    



        
    from base
),

final as (
    
    select 
        section_id,
        task_id
    from fields
)

select * 
from final