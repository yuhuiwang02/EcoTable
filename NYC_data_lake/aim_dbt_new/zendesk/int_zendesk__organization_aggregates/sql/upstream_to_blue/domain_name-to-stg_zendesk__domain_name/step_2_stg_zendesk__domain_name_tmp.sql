--To disable this model, set the using_domain_names variable within your dbt_project.yml file to False.









    select
            "index",
  "organization_id",
  "_fivetran_synced",
  "domain_name"
        from "google_ads"."public"."domain_name_data" as source_table
    
    