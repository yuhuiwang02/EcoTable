

with base as (

    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__organization_tmp"

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
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    details
    
 as 
    
    details
    
, 
    
    
    external_id
    
 as 
    
    external_id
    
, 
    
    
    group_id
    
 as 
    
    group_id
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    notes
    
 as 
    
    notes
    
, 
    
    
    shared_comments
    
 as 
    
    shared_comments
    
, 
    
    
    shared_tickets
    
 as 
    
    shared_tickets
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    
, 
    
    
    url
    
 as 
    
    url
    



        
        
, 'google_ads' || '.'|| 'public' as source_relation


    from base
),

final as (
    
    select 
        id as organization_id,
        created_at,
        updated_at,
        details,
        name,
        external_id,
        source_relation

        





    from fields
)

select * 
from final