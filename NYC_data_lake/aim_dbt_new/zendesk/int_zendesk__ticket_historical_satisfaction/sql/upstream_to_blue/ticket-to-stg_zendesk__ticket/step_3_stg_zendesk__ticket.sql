--To disable this model, set the using_ticket_form_history variable within your dbt_project.yml file to False.


with base as (

    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__ticket_form_history_tmp"

),

fields as (

    select
        /*
        The below macro is used to generate the correct SQL for package staging models. It takes a list of columns 
        that are expected/needed (staging_columns from dbt_zendesk/models/tmp/) and compares it with columns 
        in the source (source_columns from dbt_zendesk/macros/).
        For more information refer to our dbt_fivetran_utils documentation (https://github.com/fivetran/dbt_fivetran_utils.git).
        */
        
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    active
    
 as 
    
    active
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    display_name
    
 as 
    
    display_name
    
, 
    
    
    end_user_visible
    
 as 
    
    end_user_visible
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    




        
, 'google_ads' || '.'|| 'public' as source_relation


    from base
),

final as (
    
    select 
        id as ticket_form_id,
        cast(created_at as timestamp) as created_at,
        cast(updated_at as timestamp) as updated_at,
        display_name,
        active as is_active,
        name,
        source_relation
        
    from fields
    where not coalesce(_fivetran_deleted, false)
    
)

select * 
from final