-- To do -- can we delete ticket_status_counter and unique_status_counter?

with ticket_status_history as (

    select *
    from "google_ads"."public_zendesk_dev"."int_zendesk__updates"
    where field_name = 'status'

)

  select
    source_relation,
    ticket_id,
    valid_starting_at,
    valid_ending_at,
    
        (
        (
        ((coalesce(valid_ending_at, now()))::date - (valid_starting_at)::date)
     * 24 + date_part('hour', (coalesce(valid_ending_at, now()))::timestamp) - date_part('hour', (valid_starting_at)::timestamp))
     * 60 + date_part('minute', (coalesce(valid_ending_at, now()))::timestamp) - date_part('minute', (valid_starting_at)::timestamp))
     as status_duration_calendar_minutes,
    value as status,
    -- MIGHT BE ABLE TO DELETE ROWS BELOW
    row_number() over (partition by ticket_id  order by valid_starting_at) as ticket_status_counter,
    row_number() over (partition by ticket_id, value  order by valid_starting_at) as unique_status_counter

  from ticket_status_history