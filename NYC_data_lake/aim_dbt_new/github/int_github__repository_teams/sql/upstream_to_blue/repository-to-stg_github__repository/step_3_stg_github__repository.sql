with repository as (

    select *
    from "github"."public_github_dev"."stg_github__repository_tmp"

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
    cast(null as boolean) as 
    
    archived
    
 , 
    cast(null as timestamp) as 
    
    created_at
    
 , 
    cast(null as TEXT) as 
    
    default_branch
    
 , 
    cast(null as TEXT) as 
    
    description
    
 , 
    cast(null as boolean) as 
    
    fork
    
 , 
    cast(null as TEXT) as 
    
    full_name
    
 , 
    cast(null as TEXT) as 
    
    homepage
    
 , 
    cast(null as integer) as 
    
    id
    
 , 
    cast(null as TEXT) as 
    
    language
    
 , 
    cast(null as TEXT) as 
    
    name
    
 , 
    cast(null as integer) as 
    
    owner_id
    
 , 
    cast(null as boolean) as 
    
    private
    
 



    from repository

), fields as (

    select 
      id as repository_id,
      full_name,
      private as is_private

    from macro
)

select *
from fields