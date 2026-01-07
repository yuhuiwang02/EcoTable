--To disable this model, set the using_organization_tags variable within your dbt_project.yml file to False.









    select
            "organization_id",
  "tag",
  "_fivetran_synced"
        from "google_ads"."public"."organization_tag_data" as source_table
    
    