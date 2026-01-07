# Amazon Selling Partner dbt Package ([Docs](https://fivetran.github.io/dbt_amazon_selling_partner/))

<p align="left">
    <a alt="License"
        href="https://github.com/fivetran/dbt_amazon_selling_partner/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
    <a alt="dbt-core">
        <img src="https://img.shields.io/badge/dbt_Core™_version->=1.3.0_,<2.0.0-orange.svg" /></a>
    <a alt="Maintained?">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" /></a>
    <a alt="PRs">
        <img src="https://img.shields.io/badge/Contributions-welcome-blueviolet" /></a>
    <a alt="Fivetran Quickstart Compatible"
        href="https://fivetran.com/docs/transformations/dbt/quickstart">
        <img src="https://img.shields.io/badge/Fivetran_Quickstart_Compatible%3F-yes-green.svg" /></a>
</p>

## What does this dbt package do?

This package models Amazon Selling Partner data from [Fivetran's connector](https://fivetran.com/docs/applications/amazon-selling-partner). It uses data in the format described by [this ERD](https://fivetran.com/docs/applications/amazon-selling-partner#schemainformation), specifically the **ORDERS**, **CATALOG**, and **FBA** Seller Central modules.

> This package is currently not compatible with any [Vendor Central modules](https://fivetran.com/docs/connectors/applications/amazon-selling-partner#vendormodules). If you would like to see Vendor Central compatibility (or the use of any other [Seller Central modules](https://fivetran.com/docs/connectors/applications/amazon-selling-partner#sellermodules)), please open up a [Feature Request](https://github.com/fivetran/dbt_amazon_selling_partner/issues/new?template=feature-request.yml).

The main focus of the package is to transform core Seller Central object tables into analytics-ready models, including:
  - Materializes [Amazon Selling Partner staging tables](https://fivetran.github.io/dbt_amazon_selling_partner/#!/overview/amazon_selling_partner_source/models/?g_v=1) which leverage data in the format described by the **ORDERS**, **CATALOG**, and **FBA** Seller Central modules from [this ERD](https://fivetran.com/docs/applications/amazon-selling-partner/#schemainformation). These staging tables clean, test, and prepare your Amazon Selling Partner data from [Fivetran's connector](https://fivetran.com/docs/applications/amazon-selling-partner) for analysis by doing the following:
  - Names columns for consistency across all packages and for easier analysis
      - Primary keys are renamed from `_fivetran_id` to `<table name>_id`.
      - Foreign key names explicitly map onto their related tables (ie `owner_id` -> `owner_user_id`).
      - Datetime fields are renamed to `<event happened>_at`.
  - Adds column-level testing where applicable. For example, all primary keys are tested for uniqueness and non-null values.
  - Generates a comprehensive data dictionary of your Amazon Selling Partner data through the [dbt docs site](https://fivetran.github.io/dbt_amazon_selling_partner/).
  - Enables you to better analyze your Amazon Seller data by enriching the orders, order items, and listed items with catalog information and sales and current inventory aggregates.

> This package does not apply freshness tests to source data due to the variability of survey cadences.

<!--section="amazon_selling_partner_transformation_model"-->
The following table provides a detailed list of all models materialized within this package by default. 
> TIP: See more details about these models in the package's [dbt docs site](https://fivetran.github.io/dbt_amazon_selling_partner/#!/overview/amazon_selling_partner).

| **model**                 | **description**                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| [amazon_selling_partner__orders](https://fivetran.github.io/dbt_amazon_selling_partner/#!/model/model.amazon_selling_partner.amazon_selling_partner__orders)  | Table of orders placed in Amazon, enhanced with payment method information and order item aggregates.    |
| [amazon_selling_partner__order_items](https://fivetran.github.io/dbt_amazon_selling_partner/#!/model/model.amazon_selling_partner.amazon_selling_partner__order_items)  | Table of single line items of Amazon orders, enhanced with order and catalog item information.   |
| [amazon_selling_partner__item_inventory](https://fivetran.github.io/dbt_amazon_selling_partner/#!/model/model.amazon_selling_partner.amazon_selling_partner__item_inventory)  | Table containing current inventory levels pertaining to individual Amazon catalog items, enhanced with all product descriptors and identifiers, listing metadata, item dimensions, and sales ranks.   |

### Materialized Models
Each Quickstart transformation job run materializes 31 models if all components of this data model are enabled. This count includes all staging, intermediate, and final models materialized as `view`, `table`, or `incremental`.
<!--section-end-->

## How do I use the dbt package?

### Step 1: Prerequisites
To use this dbt package, you must have the following:

- At least one Fivetran Amazon Selling Partner connection syncing data into your destination.
- A **BigQuery**, **Snowflake**, **Redshift**, **Databricks**, or **PostgreSQL** destination.

### Step 2: Install the package
Include the following Amazon Selling Partner package version in your `packages.yml` file:
> TIP: Check [dbt Hub](https://hub.getdbt.com/) for the latest installation instructions or [read the dbt docs](https://docs.getdbt.com/docs/package-management) for more information on installing packages.
```yml
packages:
  - package: fivetran/amazon_selling_partner
    version: [">=0.2.0", "<0.3.0"] # we recommend using ranges to capture non-breaking changes automatically
```

### Step 3: Define database and schema variables
#### Single connection
By default, this package runs using your destination and the `amazon_selling_partner` schema. If this is not where your Amazon Selling Partner data is (for example, if your Amazon Selling Partner schema is named `amazon_selling_partner_fivetran`), add the following configuration to your root `dbt_project.yml` file:

```yml
# dbt_project.yml

vars:
    amazon_selling_partner_database: your_database_name
    amazon_selling_partner_schema: your_schema_name
```

#### Option B: Union multiple connections
If you have multiple Amazon Selling Partner connections in Fivetran and would like to use this package on all of them simultaneously, we have provided functionality to do so. For each source table, the package will union all of the data together and pass the unioned table into the transformations. The `source_relation` column in each model indicates the origin of each record.

To use this functionality, you will need to set the `amazon_selling_partner_sources` variable in your root `dbt_project.yml` file:

```yml
# dbt_project.yml

vars:
  amazon_selling_partner_sources:
    - database: connection_1_destination_name # Likely Required. Default value = target.database
      schema: connection_1_schema_name # Likely Required. Default value = 'amazon_selling_partner'
      name: connection_1_source_name # Required only if following the step in the following subsection

    - database: connection_2_destination_name
      schema: connection_2_schema_name
      name: connection_2_source_name
```

##### Recommended: Incorporate unioned sources into DAG
> *If you are running the package through [Fivetran Transformations for dbt Core™](https://fivetran.com/docs/transformations/dbt#transformationsfordbtcore), the below step is necessary in order to synchronize model runs with your Amazon Selling Partner connections. Alternatively, you may choose to run the package through Fivetran [Quickstart](https://fivetran.com/docs/transformations/quickstart), which would create separate sets of models for each Amazon Selling Partner source rather than one set of unioned models.*

<details><summary>Expand for details</summary>
<br>

By default, this package defines one single-connection source, called `amazon_selling_partner`, which will be disabled if you are unioning multiple connections. This means that your DAG will not include your Amazon Selling Partner sources, though the package will run successfully.

To properly incorporate all of your Amazon Selling Partner connections into your project's DAG:
1. Define each of your sources in a `.yml` file in your project. Utilize the following template for the `source`-level configurations, and, **most importantly**, copy and paste the table and column-level definitions from the package's `src_amazon_selling_partner.yml` [file](https://github.com/fivetran/dbt_amazon_selling_partner/blob/main/models/staging/src_amazon_selling_partner.yml).

```yml
# a .yml file in your root project
version: 2

sources:
  - name: <name> # ex: Should match name in amazon_selling_partner_sources
    schema: <schema_name>
    database: <database_name>
    loader: fivetran
    loaded_at_field: _fivetran_synced

    freshness: # feel free to adjust to your liking
      warn_after: {count: 72, period: hour}
      error_after: {count: 168, period: hour}

    tables: # copy and paste from amazon_selling_partner/models/staging/src_amazon_selling_partner.yml - see https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/ for how to use &/* anchors to only do so once
```

2. Set the `has_defined_sources` variable (scoped to the `amazon_selling_partner` package) to `True`, like such:
```yml
# dbt_project.yml
vars:
  amazon_selling_partner:
    has_defined_sources: true
```

</details>

### Step 4: Enable/Disable models for unused modules

By default, this package transforms tables from the **ORDERS**, **CATALOG**, and **FBA** [Seller Central modules](https://fivetran.com/docs/connectors/applications/amazon-selling-partner#sellermodules) described in this [ERD](https://fivetran.com/docs/connectors/applications/amazon-selling-partner#schemainformation). The package currently uses the following tables from each module:

**ORDERS** module:
- `orders`
- `order_item`
- `payment_method_detail_item`
- `order_item_promotion_id`

**CATALOG** module:
- `item_summary`
- `item_relationship`
- `item_product_type`
- `item_identifier`
- `item_display_group_sales_rank`
- `item_classification_sales_rank`
- `item_dimension`
- `item_image`

**FBA** module
- `fba_inventory_summary`
- `fba_inventory_researching_quantity_entry`

If you do not have a module enabled in your Amazon Selling Partner connection(s), you may still run the package successfully by configuring the appropriate `amazon_selling_partner__using_<module_name>_module` variable. To do so, add the following configuration to your `dbt_project.yml`:

```yml
vars:
    amazon_selling_partner__using_orders_module: False # default = True. Disables materialization of the amazon_selling_partner__orders and amazon_selling_partner__order_items models
    amazon_selling_partner__using_catalog_module: False # default = True. Disables materialization of the amazon_selling_partner__item_inventory model and removes item columns from amazon_selling_partner__order_items
    amazon_selling_partner__using_fba_module: False # default = True. Removes inventory columns from the amazon_selling_partner__item_inventory model
```

#### Quickstart
For users running the package through Fivetran [Quickstart](https://fivetran.com/docs/transformations/quickstart), these variables are dynamically assigned based on the presence of core tables in each module:
- `amazon_selling_partner__using_orders_module` is disabled if the `orders` or `order_item` source tables are missing. 
- `amazon_selling_partner__using_catalog_module` is disabled if the `item_summary` source table is missing.
- `amazon_selling_partner__using_fba_module` is disabled if the `fba_inventory_summary` source table is missing.

> If a non-core table is missing, the package will create an empty staging model with all the proper columns and data types so as to not disrupt downstream transformations.

### (Optional) Step 5: Additional configurations

#### Changing the Build Schema
By default this package will build the Amazon Selling Partner staging models within a schema titled (<target_schema> + `_stg_amazon_selling_partner`) and the Amazon Selling Partner final models within a schema titled (<target_schema> + `_amazon_selling_partner`) in your target database. If this is not where you want your modeled qualtrics data to be written to, add the following configuration to your `dbt_project.yml` file:

```yml
# dbt_project.yml

models:
  amazon_selling_partner:
    +schema: my_new_schema_name # leave blank for just the target_schema
    staging:
        +schema: my_new_schema_name # leave blank for just the target_schema
```

#### Change the source table references (available for single-connection runs only)
If an individual source table has a different name than the package expects, add the table name as it appears in your destination to the respective variable:

> IMPORTANT: See this project's [`dbt_project.yml`](https://github.com/fivetran/dbt_amazon_selling_partner/blob/main/dbt_project.yml) variable declarations to see the expected names.

```yml
# dbt_project.yml

vars:
    amazon_selling_partner_<default_source_table_name>_identifier: your_table_name 
```
</details>

### (Optional) Step 6: Orchestrate your models with Fivetran Transformations for dbt Core™
<details><summary>Expand for details</summary>
<br>

Fivetran offers the ability for you to orchestrate your dbt project through [Fivetran Transformations for dbt Core™](https://fivetran.com/docs/transformations/dbt). Learn how to set up your project for orchestration through Fivetran in our [Transformations for dbt Core setup guides](https://fivetran.com/docs/transformations/dbt#setupguide).
</details>

## Does this package have dependencies?
This dbt package is dependent on the following dbt packages. These dependencies are installed by default within this package. For more information on the following packages, refer to the [dbt hub](https://hub.getdbt.com/) site.
> IMPORTANT: If you have any of these dependent packages in your own `packages.yml` file, we highly recommend that you remove them from your root `packages.yml` to avoid package version conflicts.

```yml
packages:
    - package: fivetran/fivetran_utils
      version: [">=0.4.0", "<0.5.0"]

    - package: dbt-labs/dbt_utils
      version: [">=1.0.0", "<2.0.0"]
```

## How is this package maintained and can I contribute?
### Package Maintenance
The Fivetran team maintaining this package _only_ maintains the latest version of the package. We highly recommend you stay consistent with the [latest version](https://hub.getdbt.com/fivetran/amazon_selling_partner/latest/) of the package and refer to the [CHANGELOG](https://github.com/fivetran/dbt_amazon_selling_partner/blob/main/CHANGELOG.md) and release notes for more information on changes across versions.

### Contributions
A small team of analytics engineers at Fivetran develops these dbt packages. However, the packages are made better by community contributions.

We highly encourage and welcome contributions to this package. Check out [this dbt Discourse article](https://discourse.getdbt.com/t/contributing-to-a-dbt-package/657) on the best workflow for contributing to a package.

## Are there any resources available?
- If you have questions or want to reach out for help, refer to the [GitHub Issue](https://github.com/fivetran/dbt_amazon_selling_partner/issues/new/choose) section to find the right avenue of support for you.
- If you want to provide feedback to the dbt package team at Fivetran or want to request a new dbt package, fill out our [Feedback Form](https://www.surveymonkey.com/r/DQ7K7WW).