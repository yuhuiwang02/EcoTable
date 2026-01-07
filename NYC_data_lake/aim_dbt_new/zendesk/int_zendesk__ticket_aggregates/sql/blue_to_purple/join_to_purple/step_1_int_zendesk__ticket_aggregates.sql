with tickets as (
  select *
  from "google_ads"."public_zendesk_dev"."stg_zendesk__ticket"

), ticket_tags as (

  select *
  from "google_ads"."public_zendesk_dev"."stg_zendesk__ticket_tag"

--If you use using_brands this will be included, if not it will be ignored.

), brands as (

  select *
  from "google_ads"."public_zendesk_dev"."stg_zendesk__brand"

  
), ticket_tag_aggregate as (
  select
    source_relation,
    ticket_tags.ticket_id,
    
    string_agg(ticket_tags.tags, ', ')

 as ticket_tags
  from ticket_tags
  group by 1, 2

), final as (
  select 
    tickets.*,
    case when lower(tickets.type) = 'incident'
      then true
      else false
        end as is_incident,
    
    brands.name as ticket_brand_name,
    
    ticket_tag_aggregate.ticket_tags
  from tickets

  left join ticket_tag_aggregate
    on tickets.ticket_id = ticket_tag_aggregate.ticket_id 
    and tickets.source_relation = ticket_tag_aggregate.source_relation

  
  left join brands
    on brands.brand_id = tickets.brand_id
    and brands.source_relation = tickets.source_relation
      
)

select *
from final