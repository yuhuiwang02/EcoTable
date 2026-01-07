

with base as (

    select *
    from "mailchimp"."public_mailchimp_dev"."stg_mailchimp__automation_emails_tmp"

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
    
    
    archive_url
    
 as 
    
    archive_url
    
, 
    
    
    authenticate
    
 as 
    
    authenticate
    
, 
    
    
    auto_footer
    
 as 
    
    auto_footer
    
, 
    
    
    auto_tweet
    
 as 
    
    auto_tweet
    
, 
    
    
    automation_id
    
 as 
    
    automation_id
    
, 
    
    
    clicktale
    
 as 
    
    clicktale
    
, 
    
    
    content_type
    
 as 
    
    content_type
    
, 
    
    
    create_time
    
 as 
    
    create_time
    
, 
    
    
    delay_action
    
 as 
    
    delay_action
    
, 
    
    
    delay_action_description
    
 as 
    
    delay_action_description
    
, 
    
    
    delay_amount
    
 as 
    
    delay_amount
    
, 
    
    
    delay_direction
    
 as 
    
    delay_direction
    
, 
    
    
    delay_full_description
    
 as 
    
    delay_full_description
    
, 
    
    
    delay_type
    
 as 
    
    delay_type
    
, 
    
    
    drag_and_drop
    
 as 
    
    drag_and_drop
    
, 
    
    
    fb_comments
    
 as 
    
    fb_comments
    
, 
    
    
    folder_id
    
 as 
    
    folder_id
    
, 
    
    
    from_name
    
 as 
    
    from_name
    
, 
    
    
    google_analytics
    
 as 
    
    google_analytics
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    inline_css
    
 as 
    
    inline_css
    
, 
    
    
    position
    
 as 
    
    position
    
, 
    
    
    reply_to
    
 as 
    
    reply_to
    
, 
    
    
    send_time
    
 as 
    
    send_time
    
, 
    
    
    start_time
    
 as 
    
    start_time
    
, 
    
    
    status
    
 as 
    
    status
    
, 
    
    
    subject_line
    
 as 
    
    subject_line
    
, 
    
    
    template_id
    
 as 
    
    template_id
    
, 
    
    
    timewarp
    
 as 
    
    timewarp
    
, 
    
    
    title
    
 as 
    
    title
    
, 
    
    
    to_name
    
 as 
    
    to_name
    
, 
    
    
    track_ecomm_360
    
 as 
    
    track_ecomm_360
    
, 
    
    
    track_goals
    
 as 
    
    track_goals
    
, 
    
    
    track_html_clicks
    
 as 
    
    track_html_clicks
    
, 
    
    
    track_opens
    
 as 
    
    track_opens
    
, 
    
    
    track_text_clicks
    
 as 
    
    track_text_clicks
    
, 
    
    
    use_conversation
    
 as 
    
    use_conversation
    



        
    from base         

), 

final as (

    select
        -- IDs and standard timestamp
        id as automation_email_id,
        automation_id,
        create_time as created_timestamp,
        start_time as started_timestamp,
        send_time as send_timestamp,

        -- email details
        from_name,
        reply_to,
        status,
        subject_line,
        title,
        to_name,

        archive_url,
        authenticate,
        auto_footer,
        auto_tweet,
        clicktale,
        content_type,
        delay_action,
        delay_action_description,
        delay_amount,
        delay_direction,
        delay_full_description,
        delay_type,
        drag_and_drop,
        fb_comments,
        folder_id,
        google_analytics,
        inline_css,
        position,
        template_id,
        timewarp,
        track_ecomm_360,
        track_goals,
        track_html_clicks,
        track_opens,
        track_text_clicks,
        use_conversation
    from fields

)

select *
from final