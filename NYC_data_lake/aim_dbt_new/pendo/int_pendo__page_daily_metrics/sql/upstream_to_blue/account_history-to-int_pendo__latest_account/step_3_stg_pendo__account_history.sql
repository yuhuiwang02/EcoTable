with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__account_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    first_visit
    
 as 
    
    first_visit
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    id_hash
    
 as 
    
    id_hash
    
, 
    
    
    last_updated_at
    
 as 
    
    last_updated_at
    
, 
    
    
    last_visit
    
 as 
    
    last_visit
    



        
    from base
),

final as (
    
    select 
        id as account_id,
        last_updated_at,
        id_hash as account_id_hash,
        first_visit as first_visit_at,
        last_visit as last_visit_at,
        _fivetran_synced

        --The below macro adds the fields defined within your pendo__account_history_pass_through_columns variable into the staging model
        




        
    from fields
)

select * 
from final