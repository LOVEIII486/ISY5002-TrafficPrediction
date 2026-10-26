# LTA Traffic Images sample response

`Traffic-Imagesv2.json` and `.xml` are a sample response from the LTA DataMall
**Traffic Images** API, captured on 2020-04-14 with 87 records. The detection
pipeline uses only `CameraID`, `Latitude` and `Longitude`; see
`camera_coordinates()` in `src/detection/manifest.py`.

The signed query parameters in `ImageLink` have been removed. The original
response contained temporary AWS credentials and an expired session token; those
values should not be stored in a public repository. The retained URLs contain
only the base image path. Fresh image URLs must be obtained from the API.
