# dbt_apple_search_ads v1.0.0

[PR #35](https://github.com/fivetran/dbt_apple_search_ads/pull/35) includes the following updates:

## Breaking Changes

### Source Package Consolidation
- Removed the dependency on the `fivetran/apple_search_ads_source` package.
  - All functionality from the source package has been merged into this transformation package for improved maintainability and clarity.
  - If you reference `fivetran/apple_search_ads_source` in your `packages.yml`, you must remove this dependency to avoid conflicts.
  - Any source overrides referencing the `fivetran/apple_search_ads_source` package will also need to be removed or updated to reference this package.
  - Update any apple_search_ads_source-scoped variables to be scoped to only under this package. See the [README](https://github.com/fivetran/dbt_apple_search_ads/blob/main/README.md#change-the-build-schema) for how to configure the build schema of staging models.
- As part of the consolidation, vars are no longer used to reference staging models, and only sources are represented by vars. Staging models are now referenced directly with `ref()` in downstream models.

### dbt Fusion Compatibility Updates
- Updated package to maintain compatibility with dbt-core versions both before and after v1.10.6, which introduced a breaking change to multi-argument test syntax (e.g., `unique_combination_of_columns`).
- Temporarily removed unsupported tests to avoid errors and ensure smoother upgrades across different dbt-core versions. These tests will be reintroduced once a safe migration path is available.
  - Removed all `dbt_utils.unique_combination_of_columns` tests. 
  - Moved `loaded_at_field: _fivetran_synced` under the `config:` block in `src_apple_search_ads.yml`.

### Under the Hood
- Updated conditions in `.github/workflows/auto-release.yml`.
- Added `.github/workflows/generate-docs.yml`. 

# dbt_apple_search_ads v0.6.0

[PR #31](https://github.com/fivetran/dbt_apple_search_ads/pull/31) includes the following updates:

## Breaking Change for dbt Core < 1.9.6
> *Note: This is not relevant to Fivetran Quickstart users.*

Migrated `freshness` from a top-level source property to a source `config` in alignment with [recent updates](https://github.com/dbt-labs/dbt-core/issues/11506) from dbt Core ([Source PR #56](https://github.com/fivetran/dbt_apple_search_ads_source/pull/56)). This will resolve the following deprecation warning that users running dbt >= 1.9.6 may have received:

```
[WARNING]: Deprecated functionality
Found `freshness` as a top-level property of `apple_search_ads` in file
`models/src_apple_search_ads.yml`. The `freshness` top-level property should be moved
into the `config` of `apple_search_ads`.
```

**IMPORTANT:** Users running dbt Core < 1.9.6 will not be able to utilize freshness tests in this release or any subsequent releases, as older versions of dbt will not recognize freshness as a source `config` and therefore not run the tests.

If you are using dbt Core < 1.9.6 and want to continue running Apple Search Ads freshness tests, please elect **one** of the following options:
  1. (Recommended) Upgrade to dbt Core >= 1.9.6
  2. Do not upgrade your installed version of the `apple_search_ads` package. Pin your dependency on v0.5.0 in your `packages.yml` file.
  3. Utilize a dbt [override](https://docs.getdbt.com/reference/resource-properties/overrides) to overwrite the package's `apple_search_ads` source and apply freshness via the [old](https://github.com/fivetran/dbt_apple_search_ads_source/blob/v0.5.0/models/src_apple_search_ads.yml#L12-L14) top-level property route. This will require you to copy and paste the entirety of the `src_apple_search_ads.yml` [file](https://github.com/fivetran/dbt_apple_search_ads_source/blob/v0.5.0/models/src_apple_search_ads.yml#L5-L290) and add an `overrides: apple_search_ads_source` property.

## Under the Hood
- Updated the package maintainer PR template.

# dbt_apple_search_ads v0.5.0
[PR #29](https://github.com/fivetran/dbt_apple_search_ads/pull/29) includes the following updates:

## Breaking Changes: Apple Search Ads API v5 schema updates
- Following the connector updates surrounding the Apple Search Ads API v5 release, the following fields have been added to the upstream `*_report` staging models ([PR #54](https://github.com/fivetran/dbt_apple_search_ads_source/pull/54)) and will be used to sunset the below fields:
  - `tap_installs` - replacing `conversions`
  - `tap_new_downloads` - replacing `new_downloads`
  - `tap_redownloads` - replacing `redownloads`
- Following this, the end models now have the following new fields, which will be used to sunset the below fields:
  - `tap_new_downloads` - replacing `new_downloads`
  - `tap_redownloads` - replacing `tap_redownloads`
  - `tap_total_downloads` - replacing `total_downloads`
  - `tap_installs` - replacing `conversions`
- In order to maintain backwards compatibility and historical data, we have coalesced the new and existing versions of each field.
- **We encourage referencing the new fields, as the fields being replaced will be deprecated in May 2025 in an upcoming release as highlighted by this [ticket](https://github.com/fivetran/dbt_apple_search_ads_source/issues/55).** 

## Documentation
- Added Quickstart model counts to README. ([#28](https://github.com/fivetran/dbt_apple_search_ads/pull/28))
- Corrected references to connectors and connections in the README. ([#28](https://github.com/fivetran/dbt_apple_search_ads/pull/28))
- Updated Copyright and README format.

# dbt_apple_search_ads v0.4.1
[PR #26](https://github.com/fivetran/dbt_apple_search_ads/pull/26) includes the following updates:

## Bug Fixes
- Adjusts `inner joins` in each end model to be `left joins`. 
  - When ads and other entities are deleted from Apple Search Ads, their records are hard-deleted from the `<entity>_history` source tables, though any associated `<entity>_report` records persist with new `<entity>_id` values of `-1`. This update ensures that the report records persist into the package's end models in these cases.
>**Note**: This will likely increase the row count of your data models. To remove these newly included records, filter out rows where the `<entity>_id = -1`.
- We have accordingly made sure to select fields, specifically entity IDs, from the left side of these joins (the report tables).

# dbt_apple_search_ads v0.4.0
[PR #24](https://github.com/fivetran/dbt_apple_search_ads/pull/24) includes the following **BREAKING CHANGE** updates:

## Feature Updates: Conversion Support
- Added the `conversions` source field to each `apple_search_ads__*_report` model.
- If you are already passing in these fields via the [passthrough columns](https://github.com/fivetran/dbt_apple_search_ads?tab=readme-ov-file#passing-through-additional-metrics), the package will automatically prevent "duplicate column" errors.
> Breaking change: This update impacts users not previously including `conversions` via passthrough columns.

- Since there is no direct `conversion_value` field available in Apple Search Ads data, added a `DECISIONLOG` entry discussing alternatives. See the [Conversion Value section](https://github.com/fivetran/dbt_apple_search_ads/blob/main/DECISIONLOG.md#conversion-value) for more details.

## Under the Hood
- Introduced the `apple_search_ads_persist_pass_through_columns` macro to maintain compatibility for users already passing conversion fields through passthrough columns.
  - This macro coalesces each passthrough metric with `0`.
- Added validation tests for integrity and consistency of transformation models within the `integration_tests` folder (for maintainers only).

## Documentation
- Added missing column descriptions for `keyword_id` and `keyword_text` in the `apple_search_ads__search_term_report` model.

# dbt_apple_search_ads v0.3.0
[PR #20](https://github.com/fivetran/dbt_apple_search_ads/pull/20) includes the following updates:
## Feature update 🎉
- Unioning capability! This adds the ability to union source data from multiple apple_search_ads connectors. Refer to the [Union Multiple Connectors README section](https://github.com/fivetran/dbt_apple_search_ads/blob/main/README.md#union-multiple-connectors) for more details.

## Under the hood 🚘
- In the source package, updated tmp models to union source data using the `fivetran_utils.union_data` macro. 
- To distinguish which source each field comes from, added `source_relation` column in each staging and downstream model and applied the `fivetran_utils.source_relation` macro.
  - The `source_relation` column is included in all joins in the transform package. 
- Updated tests to account for the new `source_relation` column.

# dbt_apple_search_ads v0.2.2
## Bugfix:
- Updated the dbt_utils.unique_combination_of_columns test for the `apple_search_ads__search_term_report` to include the following fields. ([PR #18](https://github.com/fivetran/dbt_apple_search_ads/pull/18)):
        - date_day
        - search_term_text
        - keyword_id
        - ad_group_id
        - campaign_id
        - organization_id
        - match_type

## Under the Hood:

- Incorporated the new `fivetran_utils.drop_schemas_automation` macro into the end of each Buildkite integration test job. ([PR #15](https://github.com/fivetran/dbt_apple_search_ads/pull/15))
- Updated the pull request [templates](/.github). ([PR #15](https://github.com/fivetran/dbt_apple_search_ads/pull/15))

## Contributors:

- [@yuna-tang](https://github.com/yuna-tang) ([PR #17](https://github.com/fivetran/dbt_apple_search_ads/pull/17))



# dbt_apple_search_ads v0.2.1

Accidental Release

# dbt_apple_search_ads v0.2.0

## 🚨 Breaking Changes 🚨:
[PR #14](https://github.com/fivetran/dbt_apple_search_ads/pull/14) includes the following breaking changes:
- Dispatch update for dbt-utils to dbt-core cross-db macros migration. Specifically `{{ dbt_utils.<macro> }}` have been updated to `{{ dbt.<macro> }}` for the below macros:
    - `any_value`
    - `bool_or`
    - `cast_bool_to_text`
    - `concat`
    - `date_trunc`
    - `dateadd`
    - `datediff`
    - `escape_single_quotes`
    - `except`
    - `hash`
    - `intersect`
    - `last_day`
    - `length`
    - `listagg`
    - `position`
    - `replace`
    - `right`
    - `safe_cast`
    - `split_part`
    - `string_literal`
    - `type_bigint`
    - `type_float`
    - `type_int`
    - `type_numeric`
    - `type_string`
    - `type_timestamp`
    - `array_append`
    - `array_concat`
    - `array_construct`
- For `current_timestamp` and `current_timestamp_in_utc` macros, the dispatch AND the macro names have been updated to the below, respectively:
    - `dbt.current_timestamp_backcompat`
    - `dbt.current_timestamp_in_utc_backcompat`
- `packages.yml` has been updated to reflect new default `fivetran/fivetran_utils` version, previously `[">=0.3.0", "<0.4.0"]` now `[">=0.4.0", "<0.5.0"]`.

# dbt_apple_search_ads v0.1.0

## Initial Release
- This is the initial release of this package. For more information refer to the [README](/README.md).
