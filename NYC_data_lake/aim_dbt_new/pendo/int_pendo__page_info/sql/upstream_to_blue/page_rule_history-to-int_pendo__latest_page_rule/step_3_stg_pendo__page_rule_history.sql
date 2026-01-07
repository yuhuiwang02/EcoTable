with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__page_rule_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    designer_hint
    
 as 
    
    designer_hint
    
, 
    
    
    page_id
    
 as 
    
    page_id
    
, 
    
    
    page_last_updated_at
    
 as 
    
    page_last_updated_at
    
, 
    
    
    parsed_rule
    
 as 
    
    parsed_rule
    
, 
    
    
    rule
    
 as 
    
    rule
    



        
    from base
),

final as (
    
    select 

        designer_hint,
        page_id,
        page_last_updated_at,
        parsed_rule,
        rule,
        _fivetran_synced
        
    from fields
)

select * 
from final