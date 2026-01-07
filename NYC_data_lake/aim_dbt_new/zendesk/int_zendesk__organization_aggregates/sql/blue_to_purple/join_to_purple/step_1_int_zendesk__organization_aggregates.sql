

with organizations as (
    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__organization"

--If you use organization tags, this will be included, if not it will be ignored.

), organization_tags as (
    select * 
    from "google_ads"."public_zendesk_dev"."stg_zendesk__organization_tag"

), tag_aggregates as (
    select
        organizations.organization_id,
        organizations.source_relation,
        
    string_agg(organization_tags.tags, ', ')

 as organization_tags
    from organizations

    left join organization_tags
        on organizations.organization_id = organization_tags.organization_id 
        and organizations.source_relation = organization_tags.source_relation


    group by 1, 2


--If you use using_domain_names tags this will be included, if not it will be ignored.

), domain_names as (

    select *
    from "google_ads"."public_zendesk_dev"."stg_zendesk__domain_name"

), domain_aggregates as (
    select
        organizations.organization_id,
        organizations.source_relation,
        
    string_agg(domain_names.domain_name, ', ')

 as domain_names
    from organizations

    left join domain_names
        on organizations.organization_id = domain_names.organization_id 
        and organizations.source_relation = domain_names.source_relation

    group by 1, 2



), final as (
    select
        organizations.*

        --If you use organization tags this will be included, if not it will be ignored.
        
        ,tag_aggregates.organization_tags
        

        --If you use using_domain_names tags this will be included, if not it will be ignored.
        
        ,domain_aggregates.domain_names
        

    from organizations

    --If you use using_domain_names tags this will be included, if not it will be ignored.
    
    left join domain_aggregates
        on organizations.organization_id = domain_aggregates.organization_id 
        and organizations.source_relation = domain_aggregates.source_relation
    

    --If you use organization tags this will be included, if not it will be ignored.
    
    left join tag_aggregates
        on organizations.organization_id = tag_aggregates.organization_id 
        and organizations.source_relation = tag_aggregates.source_relation
    
)

select *
from final