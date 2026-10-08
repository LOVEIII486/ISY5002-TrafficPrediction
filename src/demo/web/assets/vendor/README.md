# vendor

第三方资源本地存放，无 CDN、无构建步骤。

| 文件 | 来源 | 许可 |
|---|---|---|
| `tabler.min.css` | @tabler/core 1.6.1 | MIT（见 `tabler.LICENSE`） |
| `plotly.min.js` | 已安装的 `plotly` Python 包内的 `package_data/plotly.min.js` | MIT |

`plotly.min.js` 由安装的 plotly 包直接复制而来，版本随 `requirements.txt` 里的
`plotly` 钉死。升级 plotly 后重新复制：

```bash
python -c "import plotly,pathlib,shutil; shutil.copy(pathlib.Path(plotly.__file__).parent/'package_data'/'plotly.min.js', 'src/demo/web/assets/vendor/plotly.min.js')"
```

Tabler 只 vendor 了 CSS，没有引入它的 JS —— 本控制台的交互都是自己写的，
不依赖 Tabler 的 tab / modal 组件。
