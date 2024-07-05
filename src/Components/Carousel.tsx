// src/components/Carousel.tsx
import React from 'react';
import { Carousel } from 'react-bootstrap';

interface CarouselProps {
    images: { src: string, alt: string }[];
}

const CustomCarousel: React.FC<CarouselProps> = ({ images }) => {
    return (
        <Carousel>
            {images.map((image, index) => (
                <Carousel.Item key={index}>
                    <img
                        className="d-block w-100"
                        src={image.src}
                        alt={image.alt}
                        style={{ width: '50%' }}
                    />
                </Carousel.Item>
            ))}
        </Carousel>
    );
};

export default CustomCarousel;
