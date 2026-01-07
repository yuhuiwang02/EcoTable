with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__visitor_account_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    account_id
    
 as 
    
    account_id
    
, 
    
    
    visitor_id
    
 as 
    
    visitor_id
    
, 
    
    
    visitor_last_updated_at
    
 as 
    
    visitor_last_updated_at
    



        
    from base
),

final as (
    
    select 
        account_id,
        visitor_id,
        visitor_last_updated_at,
        _fivetran_synced
        
    from fields
)

select * 
from final