

with requested_reviewer_history as (

    select *
    from "github"."public_github_dev"."stg_github__requested_reviewer_history_tmp"

), macro as (
    select
        /*
        The below macro is used to generate the correct SQL for package staging models. It takes a list of columns 
        that are expected/needed (staging_columns from dbt_github/models/tmp/) and compares it with columns 
        in the source (source_columns from dbt_github/macros/).

        For more information refer to our dbt_fivetran_utils documentation (https://github.com/fivetran/dbt_fivetran_utils.git).
        */
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as integer) as 
    
    actor_id
    
 , 
    cast(null as timestamp) as 
    
    created_at
    
 , 
    cast(null as integer) as 
    
    pull_request_id
    
 , 
    cast(null as boolean) as 
    
    removed
    
 , 
    cast(null as integer) as 
    
    requested_id
    
 



    from requested_reviewer_history

), fields as (

    select 
      pull_request_id,
      cast(created_at as timestamp) as created_at,
      requested_id,
      removed

    from macro
)

select *
from fields