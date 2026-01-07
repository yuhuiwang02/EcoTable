with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__visitor_history_tmp"

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
    
    
    last_browser_name
    
 as 
    
    last_browser_name
    
, 
    
    
    last_browser_version
    
 as 
    
    last_browser_version
    
, 
    
    
    last_operating_system
    
 as 
    
    last_operating_system
    
, 
    
    
    last_server_name
    
 as 
    
    last_server_name
    
, 
    
    
    last_updated_at
    
 as 
    
    last_updated_at
    
, 
    
    
    last_user_agent
    
 as 
    
    last_user_agent
    
, 
    
    
    last_visit
    
 as 
    
    last_visit
    
, 
    
    
    n_id
    
 as 
    
    n_id
    



        
    from base
),

final as (
    
    select 
        id as visitor_id,
        account_id,
        first_visit as first_visit_at,
        id_hash as visitor_id_hash,
        last_browser_name,
        last_browser_version,
        last_operating_system,
        last_server_name,
        last_updated_at,
        last_user_agent,
        last_visit,
        n_id,
        _fivetran_synced

        --The below macro adds the fields defined within your pendo__visitor_history_pass_through_columns variable into the staging model
        





    from fields
)

select * 
from final