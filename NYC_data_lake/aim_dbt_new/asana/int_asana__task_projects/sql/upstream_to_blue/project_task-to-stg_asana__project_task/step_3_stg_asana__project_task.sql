with base as (

    select * 
    from "asana"."public_asana_dev"."stg_asana__project_task_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    project_id
    
 as 
    
    project_id
    
, 
    
    
    task_id
    
 as 
    
    task_id
    



        
    from base
),

final as (
    
    select 
        project_id,
        task_id
    from fields
)

select * 
from final