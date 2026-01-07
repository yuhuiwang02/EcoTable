with base as (

    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__ticket_comment_tmp"

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
    cast(null as boolean) as 
    
    _fivetran_deleted
    
 , 
    
    
    body
    
 as 
    
    body
    
, 
    cast(null as integer) as 
    
    call_duration
    
 , 
    cast(null as integer) as 
    
    call_id
    
 , 
    
    
    created
    
 as 
    
    created
    
, 
    
    
    facebook_comment
    
 as 
    
    facebook_comment
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    cast(null as integer) as 
    
    location
    
 , 
    
    
    public
    
 as 
    
    public
    
, 
    cast(null as integer) as 
    
    recording_url
    
 , 
    cast(null as timestamp) as 
    
    started_at
    
 , 
    
    
    ticket_id
    
 as 
    
    ticket_id
    
, 
    cast(null as integer) as 
    
    transcription_status
    
 , 
    cast(null as integer) as 
    
    transcription_text
    
 , 
    cast(null as integer) as 
    
    trusted
    
 , 
    
    
    tweet
    
 as 
    
    tweet
    
, 
    
    
    user_id
    
 as 
    
    user_id
    
, 
    
    
    voice_comment
    
 as 
    
    voice_comment
    
, 
    cast(null as integer) as 
    
    voice_comment_transcription_visible
    
 


        
        
, 'google_ads' || '.'|| 'public' as source_relation


    from base
),

final as (
    
    select 
        id as ticket_comment_id,
        _fivetran_synced,
        _fivetran_deleted,
        body,
        cast(created as timestamp) as created_at,
        public as is_public,
        ticket_id,
        user_id,
        facebook_comment as is_facebook_comment,
        tweet as is_tweet,
        voice_comment as is_voice_comment,
        source_relation
        
    from fields
)

select * 
from final