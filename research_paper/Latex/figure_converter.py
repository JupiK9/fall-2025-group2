from PIL import Image

img = Image.open("fig/total_savings_bar_bw.png")
img.save("fig/total_savings_bar_bw.pdf")


img = Image.open("fig/savings_analysis_bw.png")
img.save("fig/savings_analysis_bw.pdf")

img = Image.open("fig/savings_by_size_total_baseline.png")
img.save("fig/savings_by_size_total_baseline.pdf")

