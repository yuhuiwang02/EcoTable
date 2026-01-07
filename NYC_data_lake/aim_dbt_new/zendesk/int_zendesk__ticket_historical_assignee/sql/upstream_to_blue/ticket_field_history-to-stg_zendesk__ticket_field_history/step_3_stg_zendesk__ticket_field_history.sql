with base as (

    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__ticket_field_history_tmp"

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
    
    
    field_name
    
 as 
    
    field_name
    
, 
    
    
    ticket_id
    
 as 
    
    ticket_id
    
, 
    
    
    updated
    
 as 
    
    updated
    
, 
    
    
    user_id
    
 as 
    
    user_id
    
, 
    
    
    value
    
 as 
    
    value
    



        
        
, 'google_ads' || '.'|| 'public' as source_relation


    from base
),

final as (
    
    select 
        ticket_id,
        field_name,
        cast(updated as timestamp) as valid_starting_at,
        cast(lead(updated) over (partition by ticket_id, field_name  order by updated) as timestamp) as valid_ending_at,
        value,
        user_id,
        source_relation
        
    from fields
)

select * 
from final