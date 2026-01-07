with pull_request_review as (

    select *
    from "github"."public_github_dev"."stg_github__pull_request_review_tmp"

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
    cast(null as TEXT) as 
    
    body
    
 , 
    cast(null as TEXT) as 
    
    commit_sha
    
 , 
    cast(null as integer) as 
    
    id
    
 , 
    cast(null as integer) as 
    
    pull_request_id
    
 , 
    cast(null as TEXT) as 
    
    state
    
 , 
    cast(null as timestamp) as 
    
    submitted_at
    
 , 
    cast(null as integer) as 
    
    user_id
    
 



    from pull_request_review

), fields as (

    select 
      id as pull_request_review_id,
      pull_request_id,
      cast(submitted_at as timestamp) as submitted_at,
      state,
      user_id

    from macro
)

select *
from fields