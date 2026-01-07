

with base as (

    select * 
    from "mailchimp"."public_mailchimp_dev"."stg_mailchimp__automations_tmp"

),

fields as (

    select
        
    
    
    _fivetran_deleted
    
 as 
    
    _fivetran_deleted
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    create_time
    
 as 
    
    create_time
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    list_id
    
 as 
    
    list_id
    
, 
    
    
    segment_id
    
 as 
    
    segment_id
    
, 
    
    
    segment_text
    
 as 
    
    segment_text
    
, 
    
    
    start_time
    
 as 
    
    start_time
    
, 
    
    
    status
    
 as 
    
    status
    
, 
    
    
    title
    
 as 
    
    title
    
, 
    
    
    trigger_settings
    
 as 
    
    trigger_settings
    



        
    from base
),

final as (

    select
        id as automation_id,
        list_id,
        segment_id, 
        segment_text,
        start_time as started_timestamp,
        create_time as created_timestamp,
        status,
        title,
        trigger_settings
    from fields

)

select *
from final