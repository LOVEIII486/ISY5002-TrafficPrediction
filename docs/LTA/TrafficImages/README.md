# LTA Traffic Images 样本响应

`Traffic-Imagesv2.json` 与 `.xml` 是 LTA DataMall「Traffic Images」接口的一份样本响应
（2020-04-14 抓取，共 87 条）。代码只用它取 `CameraID` / `Latitude` / `Longitude`
三列，见 `src/detection/manifest.py` 的 `camera_coordinates()`。

**`ImageLink` 里的签名参数已被去掉。**

原响应返回的是 S3 presigned URL，查询串里带着 LTA 的 AWS 临时访问密钥
（`AWSAccessKeyId=ASIA…`）与会话令牌。那份凭证的签名有效期是 2020-04-14，早已失效，
但它不该留在公开仓库里（会一直触发 GitHub 的 secret scanning），所以只保留了 URL 的
基础部分。

需要用这些链接的话，得自己调接口取新的。
