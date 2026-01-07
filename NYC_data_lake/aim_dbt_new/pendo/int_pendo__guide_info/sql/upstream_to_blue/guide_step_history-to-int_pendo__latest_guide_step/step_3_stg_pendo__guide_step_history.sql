with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__guide_step_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    guide_id
    
 as 
    
    guide_id
    
, 
    
    
    guide_last_updated_at
    
 as 
    
    guide_last_updated_at
    
, 
    
    
    step_id
    
 as 
    
    step_id
    



        
    from base
),

final as (
    
    select 
    
        guide_id,
        guide_last_updated_at,
        step_id,
        _fivetran_synced

    from fields
)

select * 
from final