

with base as (

    select * 
    from "github"."public_github_dev"."stg_github__team_tmp"

),

fields as (

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
    
    description
    
 , 
    cast(null as integer) as 
    
    id
    
 , 
    cast(null as TEXT) as 
    
    name
    
 , 
    cast(null as integer) as 
    
    org_id
    
 , 
    cast(null as integer) as 
    
    parent_id
    
 , 
    cast(null as TEXT) as 
    
    privacy
    
 , 
    cast(null as TEXT) as 
    
    slug
    
 


        
    from base
    
), final as (
    
    select 
        id as team_id,
        description,
        name,
        parent_id,
        privacy,
        slug
    from fields
)

select * 
from final