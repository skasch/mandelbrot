#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The mandelbrot script.

Created by Romain Mondon-Cancel on 2020-04-13 10:12:04
"""

import logging
import math
import typing as t

from PIL import Image
import numpy as np
import colorsys


class ProgressBar:

    r"""
    Create a progress bar callable in a loop.

    Args:
        prefix: Defaults to ``""``. The text to display before the progress bar.
        suffix: Defaults to ``""``. The text to display after the progress bar. You can
            use ``{progress}`` and ``{total}`` to dynamically format the suffix. E.g.:
            ``suffix="[{progress}/{total}]"``.
        decimals: Defaults to ``1``. The number of decimal to display in the percentage
            progress.
        length: Defaults to ``100``. The number of characters used for the progress bar.
        fill: Defaults to ``"█"``. The character used to fill the completed part of the
            progress bar.
        empty: Defaults to ``"-"``. The character used to fill the uncompleted part of
            the progress bar.
        print_end: Defaults to ``"\r"``. The character to send as the ``end`` argument
            to ``print`` when displaying the progress bar.

    Source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 1,
        length: int = 100,
        fill: str = "█",
        empty: str = "-",
        print_end: str = "\r",
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.empty = empty
        self.print_end = print_end

    def print(self, progress: float, total: float = 1.0) -> None:
        """
        Display progress in a loop.

        Args:
            progress: The current progress value. Progress must end when
                ``progress == total``.
            total: Defaults to ``1.0``. The total value of the progress If not provided,
                ``progress`` is expected to be a progress ratio.
        """
        percent = f"{{0:.{self.decimals}f}}".format(100 * progress / total)
        filled_length = int(self.length * progress / total)
        bars = self.fill * filled_length + self.empty * (self.length - filled_length)
        suffix = self.suffix.format(progress=progress, total=total)
        print(
            f"{self.prefix} |{bars}| {percent}% {suffix}", end=self.print_end,
        )
        if math.isclose(progress, total):
            print()


class Color(t.NamedTuple):

    """
    Represent a 8-bit encoded color without alpha channel.

    Args:
        r: The value of the red channel. An integer between ``0`` and ``255``.
        g: The value of the green channel. An integer between ``0`` and ``255``.
        b: The value of the blue channel. An integer between ``0`` and ``255``.
    """

    r: int
    g: int
    b: int

    def __iter__(self) -> t.Iterator[int]:
        """Represent the color as a ``(r, g, b)`` tuple."""
        yield from (self.r, self.g, self.b)


class ImageSize(t.NamedTuple):

    """
    Represent the size of an image in pixels.

    Args:
        width: The width of the image.
        height: The height of the image.
    """

    width: int
    height: int

    def __iter__(self) -> t.Iterator[int]:
        """Represent the image size as a ``(height, width)`` tuple."""
        yield from (self.height, self.width)


class Interval(t.NamedTuple):

    """
    Represent an interval.

    Args:
        low: The lower bound of the interval.
        high: The higher bound of the interval.
    """

    low: float
    high: float

    def size(self) -> float:
        """Represent the size of the interval."""
        return self.high - self.low


class Pixel(t.NamedTuple):

    """
    Represent a pixel.

    Args:
        row: The row index of the pixel.
        col: The column index of the pixel.
    """

    row: int
    col: int


class Mandelbrot:

    """
    Represent a Mandelbrot set.

    Args:
        image_size: The size of the image to generate.
        max_iter: The maximum number of iteration before abandoning divergence.
        colorize: A function returning a Color from a progress value, given by the ratio
            between the number of iterations before divergence and ``max_iter``.
        default_color: The color to use for pixels that did not diverge.
        center: The center of the image in the complex plane.
        min_x: The minimum value (left bound) of real values to display in the complex
            plane.
        min_y: Defaults to ``None``. The minimum value (left bound) of real values to
            display in the complex plane. If ``None``, the value will be set so that the
            ``x`` to ``y`` ratio is ``1:1``.
        print_progress: Defaults to ``True``. Whether to display the progress of the
            ``fit`` operation as a progress bar.
    """

    def __init__(
        self,
        image_size: ImageSize,
        max_iter: int,
        colorize: t.Callable[[float], Color],
        default_color: Color,
        center: complex,
        min_x: float,
        min_y: t.Optional[float] = None,
        print_progress: bool = True,
    ):
        self.image_size = image_size
        self.center = center
        self.min_x = min_x
        self.min_y = min_y
        self.max_iter = max_iter
        self.colorize = colorize
        self.default_color = default_color
        self.print_progress = print_progress

        self.values: t.Optional[np.ndarray] = None
        self.progress_bar = ProgressBar(
            prefix=self.__class__.__name__, suffix="[{progress}/{total}]"
        )

    @property
    def x_interval(self) -> Interval:
        """Represent the interval of values along the ``x`` axis."""
        return Interval(self.min_x, self.min_x + 2 * (self.center.real - self.min_x))

    @property
    def y_interval(self) -> Interval:
        """Represent the interval of values along the ``y`` axis."""
        if self.min_y is None:
            image_ratio = self.image_size.height / self.image_size.width
            min_y = self.center.imag - image_ratio * (self.center.real - self.min_x)
        else:
            min_y = self.min_y
        return Interval(min_y, min_y + 2 * (self.center.imag - min_y))

    def compute(self, c0: complex) -> float:
        """
        Compute the divergence ratio for a given initial point ``c0``.

        The Mandelbrot is given by the divergence of the sequence ``c_0 = c0``,
        ``c_{n+1} = c_n^2 + c_0``. If ``|c_n| > 2`` at any point, the sequence will
        diverge. This function returns the ratio between the number of iterations before
        ``|c_n| > 2`` and ``self.max_iter``. If the sequence doesn't diverge before
        that, the function returns ``np.NaN``.

        Args:
            c0: The initial point of the computation.

        Returns:
            ``n / self.max_iter`` where ``n`` is the number of iterations before
            divergence if the value diverges before that, ``np.NaN`` otherwise.
        """
        c: complex = c0
        for i in range(self.max_iter):
            if abs(c) > 2:
                return i / self.max_iter
            c = c ** 2 + c0
        return np.NaN

    def ratio_color(self, ratio: float) -> Color:
        """
        Return the color corresponding to a convergence ratio ``ratio``.

        Args:
            ratio: The convergence ratio of a given point; if ``np.NaN``, the point is
                not considered converged.

        Returns:
            A ``Color`` object giving the color of that ``ratio``.
        """
        return self.default_color if np.isnan(ratio) else self.colorize(ratio)

    def colors(self) -> np.ndarray:
        """
        Return the array of colors for a fitted ``Mandelbrot`` instance.

        Raises:
            AttributeError: If the ``Mandelbrot`` instance hasn't been fitted yet.

        Returns:
            The color values of all the pixels for the Mandelbrot image.
        """
        if self.values is None:
            raise AttributeError(
                "Mandelbrot set hasn't been fit yet; please call the `fit` method "
                "before extracting the image."
            )
        return np.vectorize(
            lambda ratio: np.array(self.ratio_color(ratio)), signature="()->(n)"
        )(self.values).astype("uint8")

    def pixel_to_complex(self, pixel: Pixel) -> complex:
        """
        Return the complex number corresponding to a given pixel.

        Args:
            pixel: A ``Pixel`` object representing the coordinates of a pixel in the
                image.

        Returns:
            The corresponding complex number at that position.
        """
        relative_x = pixel.col / self.image_size.width
        relative_y = pixel.row / self.image_size.height
        x = self.x_interval.low + relative_x * self.x_interval.size()
        y = self.y_interval.high - relative_y * self.y_interval.size()
        return complex(x, y)

    def compute_pixel(self, pixel: Pixel) -> float:
        """
        Return the convergence ratio of a pixel.

        Args:
            pixel: A ``Pixel`` object representing the coordinates of a pixel in the
                image.

        Returns:
            The convergence ratio, given by the ``compute`` method, corresponding to
            ``pixel``.
        """
        c0 = self.pixel_to_complex(pixel)
        return self.compute(c0)

    def fit(self, force: bool = False) -> "Mandelbrot":
        """
        Fit a ``Mandelbrot`` instance.

        Args:
            force: Defaults to ``False``. If ``True``, overwrite the already fitted
                values.

        Returns:
            The fitted ``Mandelbrot`` instance.
        """
        if self.values is not None and not force:
            logging.warning("Tried to re-fit already generated mandelbrot set.")
        self.values = np.empty(tuple(self.image_size))
        for col in range(self.image_size.width):
            if self.print_progress:
                self.progress_bar.print(col, self.image_size.width)
            for row in range(self.image_size.height):
                self.values[row, col] = self.compute_pixel(Pixel(row, col))
        if self.print_progress:
            self.progress_bar.print(self.image_size.width, self.image_size.width)
        return self

    def image(self) -> Image:
        """
        Return an image object representing the ``Mandelbrot`` fitted instance.

        Returns:
            A ``PIL.Image`` instance representing the ``Mandelbrot`` set.
        """
        return Image.fromarray(self.colors())

    def save(self, filepath: str) -> "Mandelbrot":
        """
        Save the instance as an image file at the ``filepath`` location.

        Args:
            filepath: The path to an image file where to save the image of the
                ``Mandelbrot`` instance.

        Returns:
            The ``Mandelbrot`` instance itself.
        """
        img = self.image()
        img.save(filepath)
        return self


class Julia(Mandelbrot):

    """
    Represent a Julia set.

    Args:
        c: The complex point to build the Julia set for.
        image_size: The size of the image to generate.
        max_iter: The maximum number of iteration before abandoning divergence.
        colorize: A function returning a Color from a progress value, given by the ratio
            between the number of iterations before divergence and ``max_iter``.
        default_color: The color to use for pixels that did not diverge.
        center: The center of the image in the complex plane.
        min_x: The minimum value (left bound) of real values to display in the complex
            plane.
        min_y: Defaults to ``None``. The minimum value (left bound) of real values to
            display in the complex plane. If ``None``, the value will be set so that the
            ``x`` to ``y`` ratio is ``1:1``.
        print_progress: Defaults to ``True``. Whether to display the progress of the
            ``fit`` operation as a progress bar.
    """

    def __init__(self, c: complex, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self.c = c

    def R(self) -> float:
        """
        Represent the horizon of the Julia set.

        The horizon is the norm over which we know the values of the sequence will
        diverge. This value is the solution of the equation ``R^2 - R >= |c|``.

        Returns:
            The horizon of the Julia set.
        """
        return 0.5 * (1 + (1 + 4 * abs(self.c)) ** 0.5)

    def compute(self, z: complex) -> float:
        """
        Compute the divergence ratio for the ``Julia`` instance.

        The Julia set is given by the divergence of the sequence ``z_0 = z``,
        ``z_{n+1} = z_n^2 + c``. If ``|c_n| > self.R`` at any point, the sequence will
        diverge. This function returns the ratio between the number of iterations before
        ``|c_n| > self.R`` and ``self.max_iter``. If the sequence doesn't diverge before
        that, the function returns ``np.NaN``.

        Args:
            z: a complex number to compute the divergence ratio for.

        Returns:
            ``n / self.max_iter`` where ``n`` is the number of iterations before
            divergence if the value diverges before that, ``np.NaN`` otherwise.
        """
        c: complex = self.c
        for i in range(self.max_iter):
            if abs(z) > self.R():
                return i / self.max_iter
            z = z ** 2 + c
        return np.NaN


def colorize(ratio: float) -> Color:
    """
    Convet a ratio to a nice color to display the Mandelbrot set.

    Args:
        ratio: The ratio to convert to a ``Color``.

    Returns:
        The color corresponding to ``ratio``.
    """
    r = ratio ** 0.5
    color = 255 * np.array(
        colorsys.hsv_to_rgb((0.45 + 0.6 * r) % 1.0, 0.3 + 0.6 * r, 0.9 + 0.1 * r)
    )
    return Color(*color.astype(int))


if __name__ == "__main__":
    mandelbrot = Mandelbrot(
        image_size=ImageSize(1920 // 4, 1080 // 4),
        center=complex(-1.315, 0.1),
        min_x=-1.4,
        max_iter=2000,
        colorize=colorize,
        default_color=Color(0, 0, 0),
    )
    # mandelbrot.fit()
    # mandelbrot.save("output/img/mandelbrot.png")
    julia = Julia(
        c=complex(-0.7269, 0.1889),
        image_size=ImageSize(1920, 1080),
        center=complex(0.25, 0.26),
        min_x=0.05,
        max_iter=2000,
        colorize=colorize,
        default_color=Color(255, 255, 255),
    )
    julia.fit()
    julia.save("output/img/julia.png")
