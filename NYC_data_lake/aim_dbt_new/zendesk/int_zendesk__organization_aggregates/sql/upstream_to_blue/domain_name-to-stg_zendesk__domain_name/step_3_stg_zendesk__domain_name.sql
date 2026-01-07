--To disable this model, set the using_domain_names variable within your dbt_project.yml file to False.


with base as (

    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__domain_name_tmp"

),

fields as (

    select
        /*
        The below macro is used to generate the correct SQL for package staging models. It takes a list of columns 
        that are expected/needed (staging_columns from dbt_zendesk/models/tmp/) and compares it with columns 
        in the source (source_columns from dbt_zendesk/macros/).
        For more information refer to our dbt_fivetran_utils documentation (https://github.com/fivetran/dbt_fivetran_utils.git).
        */
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    domain_name
    
 as 
    
    domain_name
    
, 
    
    
    index
    
 as 
    
    index
    
, 
    
    
    organization_id
    
 as 
    
    organization_id
    




        
, 'google_ads' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        organization_id,
        domain_name,
        index,
        source_relation
        
    from fields
)

select * 
from final